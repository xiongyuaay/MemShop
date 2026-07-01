from __future__ import annotations

# Fill this value directly before running. This script intentionally does not
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

from retail_thor.fbx_navigation_run import (  # noqa: E402
    apply_navigation_action,
    default_initial_navigation_state,
    render_navigation_observation,
)
from retail_thor.npc import CustomerNPC, GPT4oMiniNPCPlanner  # noqa: E402
from retail_thor.navigation_agent import NAVIGATION_ACTIONS  # noqa: E402
from retail_thor.world_knowledge_manager import WorldKnowledgeManager  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHELF_FBX = ROOT.parent / "store-shelves" / "source" / "grocery_shelf.fbx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpt4o_mini_customer_npc_task"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run task 2: GPT-4o-mini CustomerNPC clarification plus navigation.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shelf-fbx", default=str(DEFAULT_SHELF_FBX))
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    client = create_openai_client()
    report = run_customer_npc_task(
        client=client,
        output_dir=Path(args.output_dir),
        shelf_fbx=Path(args.shelf_fbx),
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "success": report["success"],
                "termination": report["termination"],
                "turns": len(report["dialogue"]),
                "navigation_steps": len(report["navigation_trace"]),
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def create_openai_client():
    if OPENAI_API_KEY.startswith("PASTE_"):
        raise RuntimeError("Fill OPENAI_API_KEY at the top of this script before running task 2.")
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


def run_customer_npc_task(
    client: Any,
    output_dir: Path,
    shelf_fbx: Path = DEFAULT_SHELF_FBX,
    width: int = 900,
    height: int = 600,
    max_steps: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    raw_responses_dir = output_dir / "raw_responses"
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_task2_scenario()
    wkm = WorldKnowledgeManager(scenario["catalog"])
    npc = CustomerNPC(
        initial_request=scenario["npc_initial_request"],
        target_product_id=scenario["hidden_target_product_id"],
        wkm=wkm,
        planner=GPT4oMiniNPCPlanner(model=OPENAI_MODEL, client=client, max_output_tokens=MAX_OUTPUT_TOKENS),
    )

    dialogue: list[dict[str, Any]] = []
    initial_utterance = npc.initial_utterance()
    dialogue.append({"speaker": "npc", "utterance": initial_utterance, "intent": "initial_ambiguous_request"})
    agent_decision_trace: list[dict[str, Any]] = []
    navigation_trace: list[dict[str, Any]] = []
    state = default_initial_navigation_state(shelf_fbx)
    done = False

    try:
        for step_idx in range(max_steps):
            input_image = images_dir / f"step_{step_idx:03d}_input.png"
            observation = render_navigation_observation(shelf_fbx, input_image, state, width=width, height=height)
            agent_prompt = build_agent_task2_prompt(
                npc_initial_request=initial_utterance,
                dialogue=dialogue,
                visible_products=scenario["visible_products_for_agent"],
                agent_decision_trace=agent_decision_trace,
                navigation_trace=navigation_trace,
            )
            agent_response = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": agent_prompt},
                            {"type": "input_image", "image_url": _image_to_data_url(input_image)},
                        ],
                    }
                ],
                text={"format": _agent_task2_decision_json_schema()},
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            agent_raw_text = _response_text(agent_response)
            agent_decision = _loads_json_object(agent_raw_text)
            agent_response_path = raw_responses_dir / f"agent_step_{step_idx:03d}_response.json"
            agent_response_path.write_text(_response_to_json(agent_response, agent_raw_text), encoding="utf-8")
            decision_type = str(agent_decision.get("decision_type", ""))
            decision_record = {
                "step_idx": step_idx,
                "decision_type": decision_type,
                "utterance": str(agent_decision.get("utterance", "")),
                "action": str(agent_decision.get("action", "")),
                "rationale": str(agent_decision.get("rationale", "")),
                "confidence": float(agent_decision.get("confidence", 0.0)),
                "input": {
                    "prompt": agent_prompt,
                    "image_path": str(input_image.relative_to(output_dir)),
                    "visible_products": scenario["visible_products_for_agent"],
                    "dialogue": list(dialogue),
                    "navigation_trace": list(navigation_trace),
                },
                "raw_response_path": str(agent_response_path.relative_to(output_dir)),
                "observation": observation,
            }

            if decision_type == "ask_npc":
                decision_record["action"] = ""
                agent_utterance = decision_record["utterance"]
                dialogue.append(
                    {
                        "speaker": "agent",
                        "utterance": agent_utterance,
                        "intent": "clarify_target_attribute",
                        "rationale": decision_record["rationale"],
                        "raw_response_path": decision_record["raw_response_path"],
                    }
                )
                npc_response = npc.respond(agent_utterance)
                npc_response_path = raw_responses_dir / f"npc_step_{step_idx:03d}_response.json"
                npc_response_path.write_text(
                    json.dumps(
                        {
                            "utterance": npc_response.utterance,
                            "rationale": npc_response.rationale,
                            "product_id": npc_response.product_id,
                            "wkm_calls": npc_response.wkm_calls or [],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                dialogue.append(
                    {
                        "speaker": "npc",
                        "utterance": npc_response.utterance,
                        "intent": "answer_clarification",
                        "rationale": npc_response.rationale,
                        "product_id": npc_response.product_id,
                        "wkm_calls": npc_response.wkm_calls or [],
                        "raw_response_path": str(npc_response_path.relative_to(output_dir)),
                    }
                )
                decision_record["npc_response"] = dialogue[-1]
            elif decision_type == "navigation_action":
                action = decision_record["action"]
                state_before = state
                state_after = apply_navigation_action(state, action)
                nav_record = {
                    "step_idx": step_idx,
                    "action": action,
                    "rationale": decision_record["rationale"],
                    "confidence": decision_record["confidence"],
                    "image_path": str(input_image.relative_to(output_dir)),
                    "state_before": state_before.to_dict(),
                    "state_after": state_after.to_dict(),
                }
                navigation_trace.append(nav_record)
                state = state_after
                if action == "Done":
                    done = True
                    agent_decision_trace.append(decision_record)
                    break
            else:
                raise ValueError(f"Unsupported task2 agent decision_type: {decision_type!r}")

            agent_decision_trace.append(decision_record)
    except Exception as exc:
        return _write_task2_report(
            output_dir=output_dir,
            scenario=scenario,
            dialogue=dialogue,
            agent_decision_trace=agent_decision_trace,
            navigation_trace=navigation_trace,
            success=False,
            termination={
                "status": "error",
                "reason": "runtime_exception",
                "done_action_required": True,
                "error_message": f"{type(exc).__name__}: {exc}",
            },
        )

    return _write_task2_report(
        output_dir=output_dir,
        scenario=scenario,
        dialogue=dialogue,
        agent_decision_trace=agent_decision_trace,
        navigation_trace=navigation_trace,
        success=done,
        termination={
            "status": "done" if done else "max_steps_exceeded",
            "reason": "embodied_agent_done_after_clarification" if done else "agent_did_not_emit_done",
            "done_action_required": True,
        },
    )


def _write_task2_report(
    *,
    output_dir: Path,
    scenario: dict[str, Any],
    dialogue: list[dict[str, Any]],
    agent_decision_trace: list[dict[str, Any]],
    navigation_trace: list[dict[str, Any]],
    success: bool,
    termination: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "run_type": "real_gpt4o_mini_customer_npc_navigation_task",
        "model": OPENAI_MODEL,
        "proxy": {"http": CLASH_HTTP_PROXY, "https": CLASH_HTTPS_PROXY},
        "images_dir": "images",
        "scenario": scenario_without_catalog(scenario),
        "dialogue": dialogue,
        "agent_decision_trace": agent_decision_trace,
        "navigation_trace": navigation_trace,
        "success": success,
        "termination": termination,
    }
    report_path = output_dir / "task2_customer_npc_run.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_task2_scenario() -> dict[str, Any]:
    catalog = [
        {
            "product_id": "shape_red_cup_cylinder_001",
            "display_name_en": "red cup-shaped cylinder",
            "display_name_zh": "红色杯状圆柱体",
            "category": "shape_target",
            "target_shape": "cup-shaped cylinder with handle",
            "attributes": ["red", "cup-shaped", "cylindrical", "container-like", "handle"],
            "brand": "Shape Set",
            "price": 12,
            "stock_status": "in_stock",
            "shelf_id": "fbx_shelf_middle",
            "visual_description": (
                "A red cup-shaped cylindrical item with a small handle; visually distinct from red round items."
            ),
            "approx_location": "middle shelves; look for a red cylinder/cup shape rather than a red sphere",
        },
        {
            "product_id": "shape_red_sphere_a_001",
            "display_name_en": "red round sphere A",
            "display_name_zh": "红色圆球A",
            "category": "shape_candidate",
            "target_shape": "round sphere",
            "attributes": ["red", "round", "spherical"],
            "brand": "Shape Set",
            "price": 6,
            "stock_status": "in_stock",
            "shelf_id": "fbx_shelf_upper",
            "visual_description": "A red round spherical item.",
            "approx_location": "among the round red products on the shelf",
        },
        {
            "product_id": "shape_red_sphere_b_001",
            "display_name_en": "red round sphere B",
            "display_name_zh": "红色圆球B",
            "category": "shape_candidate",
            "target_shape": "round sphere",
            "attributes": ["red", "round", "spherical"],
            "brand": "Shape Set",
            "price": 5,
            "stock_status": "in_stock",
            "shelf_id": "fbx_shelf_middle",
            "visual_description": "Another red round spherical item, visually similar to red round sphere A.",
            "approx_location": "near other round red products on the shelf",
        },
        {
            "product_id": "shape_brown_oval_001",
            "display_name_en": "brown oval item",
            "display_name_zh": "棕色椭圆体",
            "category": "shape_candidate",
            "target_shape": "oval",
            "attributes": ["brown", "oval"],
            "brand": "Shape Set",
            "price": 8,
            "stock_status": "in_stock",
            "shelf_id": "fbx_shelf_lower",
            "visual_description": "A brown oval item.",
            "approx_location": "lower shelf area",
        },
    ]
    return {
        "npc_initial_request": "I want a red product from the shelf, but I only remember its shape.",
        "hidden_target_product_id": "shape_red_cup_cylinder_001",
        "ambiguity": "The request could match multiple red products with different shapes.",
        "catalog": catalog,
        "visible_products_for_agent": [
            _visible_product_summary(product)
            for product in catalog
            if product["product_id"] in {
                "shape_red_cup_cylinder_001",
                "shape_red_sphere_a_001",
                "shape_red_sphere_b_001",
            }
        ],
    }


def scenario_without_catalog(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "npc_initial_request": scenario["npc_initial_request"],
        "hidden_target_product_id": scenario["hidden_target_product_id"],
        "ambiguity": scenario["ambiguity"],
        "visible_products_for_agent": scenario["visible_products_for_agent"],
    }


def build_agent_task2_prompt(
    npc_initial_request: str,
    dialogue: list[dict[str, Any]],
    visible_products: list[dict[str, Any]],
    agent_decision_trace: list[dict[str, Any]],
    navigation_trace: list[dict[str, Any]],
) -> str:
    recent_action = navigation_trace[-1]["action"] if navigation_trace else ""
    repeated_action_count = 0
    for item in reversed(navigation_trace):
        if item.get("action") != recent_action:
            break
        repeated_action_count += 1
    payload = {
        "task": "customer_npc_clarification_then_navigation",
        "npc_initial_request": npc_initial_request,
        "dialogue": dialogue,
        "visible_products": visible_products,
        "agent_decision_trace": agent_decision_trace,
        "navigation_trace": navigation_trace,
        "recent_navigation_repetition": {
            "last_action": recent_action,
            "consecutive_count": repeated_action_count,
        },
        "allowed_decision_types": ["ask_npc", "navigation_action"],
        "allowed_navigation_actions": list(NAVIGATION_ACTIONS),
        "termination_policy": {
            "done_when": (
                "If the clarified target is already clearly visible, visually identifiable, and reasonably centered "
                "or large enough in the image, choose navigation_action with action=Done."
            ),
            "not_required": "This is a find-and-confirm navigation task, not a pickup task; no pickup or physical interaction is required.",
            "avoid_loop": (
                "Do not repeat MoveAhead more than two consecutive navigation steps unless the target is visibly "
                "getting larger and still not clear enough. If repeated MoveAhead no longer improves the view, use "
                "RotateLeft, RotateRight, LookUp, LookDown, Done, or ask_npc depending on ambiguity."
            ),
        },
        "instruction": (
            "You are the embodied agent planner for task 2. At any step, inspect the current image, "
            "the NPC request, the dialogue history, and the navigation history. If the target is ambiguous "
            "from the current image and dialogue, choose decision_type=ask_npc and ask exactly one concise "
            "clarification question. If the target is sufficiently clarified, choose decision_type=navigation_action "
            "and output one navigation action. For ask_npc, set action to an empty string. For navigation_action, "
            "set utterance to an empty string unless a short status message is necessary. Use Done as soon as the "
            "clarified target product is clearly visible and close or centered enough under termination_policy. "
            "NPC is separate from the embodied agent and never outputs actions."
        ),
    }
    return (
        "Generate the embodied agent's next decision for an NPC interaction and navigation task. "
        "Return JSON only.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_agent_clarification_prompt(npc_initial_request: str, visible_products: list[dict[str, Any]]) -> str:
    return build_agent_task2_prompt(
        npc_initial_request=npc_initial_request,
        dialogue=[{"speaker": "npc", "utterance": npc_initial_request}],
        visible_products=visible_products,
        agent_decision_trace=[],
        navigation_trace=[],
    )


def _visible_product_summary(product: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "display_name_en": product["display_name_en"],
        "display_name_zh": product["display_name_zh"],
        "category": product["category"],
        "attributes": product["attributes"],
        "shelf_id": product["shelf_id"],
    }
    for key in ("target_shape", "visual_description", "approx_location"):
        if key in product:
            summary[key] = product[key]
    return summary


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


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("model response must be a JSON object")
    return data


def _agent_task2_decision_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "agent_task2_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision_type": {"type": "string", "enum": ["ask_npc", "navigation_action"]},
                "utterance": {"type": "string"},
                "action": {"type": "string", "enum": ["", *NAVIGATION_ACTIONS]},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["decision_type", "utterance", "action", "rationale", "confidence"],
        },
    }


def _agent_question_json_schema() -> dict[str, Any]:
    return _agent_task2_decision_json_schema()


if __name__ == "__main__":
    raise SystemExit(main())
