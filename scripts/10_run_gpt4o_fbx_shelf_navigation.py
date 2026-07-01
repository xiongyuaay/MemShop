from __future__ import annotations

# Fill these values directly before running. This script intentionally does not
# read the API key from environment variables.
OPENAI_API_KEY = "sk-HwRnWQT9DaRYDQWGxzUTMaqqPHmPQFTxJJSpl5WpQWe7xMfC"
OPENAI_BASE_URL = "https://xiaoai.plus/v1"
OPENAI_MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 220

# clash_on proxy used on this machine.
CLASH_HTTP_PROXY = "http://127.0.0.1:10808"
CLASH_HTTPS_PROXY = "http://127.0.0.1:10808"
REQUEST_TIMEOUT_SECONDS = 90.0

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _bootstrap  # noqa: E402,F401
from PIL import Image, ImageDraw  # noqa: E402

from retail_thor.fbx_navigation_run import (  # noqa: E402
    apply_navigation_action,
    build_navigation_prompt,
    default_initial_navigation_state,
    render_navigation_observation,
)
from retail_thor.navigation_agent import NAVIGATION_ACTIONS  # noqa: E402
from retail_thor.vlm_navigation_brain import parse_navigation_decision  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHELF_FBX = ROOT.parent / "store-shelves" / "source" / "grocery_shelf.fbx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpt4o_fbx_blue_cylinder_run"
DEFAULT_INSTRUCTION = "找到蓝色柱形商品，较细的。三个中任意一个都可以。"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real GPT-4o visual navigation loop on the FBX shelf scene.")
    parser.add_argument("--shelf-fbx", default=str(DEFAULT_SHELF_FBX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    client = create_openai_client()
    report = run_navigation(
        client=client,
        shelf_fbx=Path(args.shelf_fbx),
        output_dir=Path(args.output_dir),
        instruction=args.instruction,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({"steps": len(report["trajectory"]), "success": report["success"], "report_path": report["report_path"]}, ensure_ascii=False, indent=2))
    return 0


def create_openai_client():
    if OPENAI_API_KEY.startswith("PASTE_"):
        raise RuntimeError("Fill OPENAI_API_KEY at the top of this script before running a real GPT-4o trajectory.")
    configure_hardcoded_proxy()
    try:
        import httpx
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies first, e.g. `pip install -r requirements.txt`.") from exc
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        http_client=_make_httpx_client(httpx),
    )


def configure_hardcoded_proxy() -> None:
    os.environ["HTTP_PROXY"] = CLASH_HTTP_PROXY
    os.environ["HTTPS_PROXY"] = CLASH_HTTPS_PROXY
    os.environ["http_proxy"] = CLASH_HTTP_PROXY
    os.environ["https_proxy"] = CLASH_HTTPS_PROXY
    os.environ["ALL_PROXY"] = CLASH_HTTP_PROXY
    os.environ["all_proxy"] = CLASH_HTTP_PROXY


def _make_httpx_client(httpx_module):
    proxy_attempts = [
        {"proxy": CLASH_HTTP_PROXY},
        {"proxies": CLASH_HTTP_PROXY},
        {"proxies": {"http://": CLASH_HTTP_PROXY, "https://": CLASH_HTTPS_PROXY}},
    ]
    last_error: TypeError | None = None
    for proxy_kwargs in proxy_attempts:
        try:
            return httpx_module.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True, **proxy_kwargs)
        except TypeError as exc:
            last_error = exc
    raise RuntimeError("Installed httpx version does not accept known proxy configuration styles.") from last_error


def run_navigation(
    client: Any,
    shelf_fbx: Path,
    output_dir: Path,
    instruction: str,
    max_steps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    raw_responses_dir = output_dir / "raw_responses"
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    state = default_initial_navigation_state(shelf_fbx)
    history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    scene_context = {
        "scene": "standalone_store_shelf_fbx",
        "shelf_asset": str(shelf_fbx),
        "target": "blue thin cylindrical product; any one of the visible candidates is acceptable",
    }

    for step_idx in range(max_steps):
        input_image = images_dir / f"step_{step_idx:03d}_input.png"
        observation = render_navigation_observation(shelf_fbx, input_image, state, width=width, height=height)
        prompt = build_navigation_prompt(instruction, history, scene_context)
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _image_to_data_url(input_image)},
                    ],
                }
            ],
            text={"format": _navigation_json_schema()},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        raw_text = _response_text(response)
        decision = parse_navigation_decision(raw_text)
        raw_response_path = raw_responses_dir / f"step_{step_idx:03d}_response.json"
        raw_response_path.write_text(_response_to_json(response, raw_text), encoding="utf-8")

        state_before = state
        state_after = apply_navigation_action(state, decision.action)
        annotated_image = images_dir / f"step_{step_idx:03d}_annotated.png"
        _write_annotated_image(input_image, annotated_image, step_idx, decision.action, decision.rationale)

        step_record = {
            "step_idx": step_idx,
            "input": {
                "model": OPENAI_MODEL,
                "prompt": prompt,
                "image_path": str(input_image.relative_to(output_dir)),
                "allowed_actions": list(NAVIGATION_ACTIONS),
            },
            "observation": observation,
            "raw_response_path": str(raw_response_path.relative_to(output_dir)),
            "model_output": {
                "action": decision.action,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
                "raw_text": raw_text,
            },
            "executed_action": decision.action,
            "state_before": state_before.to_dict(),
            "state_after": state_after.to_dict(),
            "annotated_image_path": str(annotated_image.relative_to(output_dir)),
            "success": decision.action == "Done",
        }
        trajectory.append(step_record)
        history.append(
            {
                "step_idx": step_idx,
                "action": decision.action,
                "success": True,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
            }
        )
        state = state_after
        if decision.action == "Done":
            break

    report = {
        "run_type": "real_gpt4o_api_call",
        "model": OPENAI_MODEL,
        "proxy": {
            "http": CLASH_HTTP_PROXY,
            "https": CLASH_HTTPS_PROXY,
        },
        "instruction": instruction,
        "shelf_fbx": str(shelf_fbx),
        "allowed_actions": list(NAVIGATION_ACTIONS),
        "success": bool(trajectory and trajectory[-1]["executed_action"] == "Done"),
        "trajectory": trajectory,
    }
    report_path = output_dir / "gpt4o_blue_cylinder_run.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise ValueError("OpenAI response does not contain output_text")


def _response_to_json(response: Any, raw_text: str) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    return json.dumps({"output_text": raw_text}, ensure_ascii=False, indent=2)


def _navigation_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "navigation_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": list(NAVIGATION_ACTIONS)},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["action", "rationale", "confidence"],
        },
    }


def _write_annotated_image(source: Path, output: Path, step_idx: int, action: str, rationale: str) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    overlay_height = 72
    draw.rectangle((0, 0, image.width, overlay_height), fill=(18, 24, 30))
    draw.text((14, 12), f"step {step_idx:03d} | action: {action}", fill=(245, 248, 250))
    rationale_line = rationale[:150] + ("..." if len(rationale) > 150 else "")
    draw.text((14, 38), rationale_line, fill=(210, 220, 230))
    image.save(output)


if __name__ == "__main__":
    raise SystemExit(main())
