from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _bootstrap  # noqa: E402,F401

from retail_thor.fbx_navigation_run import (  # noqa: E402
    apply_navigation_action,
    default_initial_navigation_state,
    render_navigation_observation,
)
from retail_thor.fbx_shelf_scene import AI2THOR_COMMON_PRODUCTS, load_store_shelf_meshes  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHELF_FBX = ROOT.parent / "store-shelves" / "source" / "grocery_shelf.fbx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpt4o_mini_product_manipulation_task"
DEFAULT_INSTRUCTION = "Move the red mug from its source shelf position to the destination shelf position."
TARGET_RENDERER_PRODUCT_INDEX = 3
PICKUP_DISTANCE_THRESHOLD = 95.0

MANIPULATION_ACTIONS = (
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Done",
    "AskQuestion",
    "PickupObject",
    "DropObject",
)
NAVIGATION_ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Done")


def _load_task2_openai_config() -> tuple[str, str, str, str, str, int]:
    task2_script = SCRIPT_DIR / "11_run_gpt4o_mini_fbx_customer_npc_task.py"
    spec = importlib.util.spec_from_file_location("_task2_config_for_task3", task2_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OpenAI config from {task2_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.OPENAI_API_KEY,
        module.OPENAI_BASE_URL,
        "gpt-4o-mini",
        module.CLASH_HTTP_PROXY,
        module.CLASH_HTTPS_PROXY,
        module.REQUEST_TIMEOUT_SECONDS,
    )


OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, CLASH_HTTP_PROXY, CLASH_HTTPS_PROXY, REQUEST_TIMEOUT_SECONDS = (
    _load_task2_openai_config()
)
MAX_OUTPUT_TOKENS = 240


def main() -> int:
    parser = argparse.ArgumentParser(description="Run task 3: GPT-4o-mini product manipulation on the FBX shelf scene.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shelf-fbx", default=str(DEFAULT_SHELF_FBX))
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    client = create_openai_client()
    report = run_product_manipulation_task(
        client=client,
        output_dir=Path(args.output_dir),
        shelf_fbx=Path(args.shelf_fbx),
        instruction=args.instruction,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "success": report["success"],
                "termination": report["termination"],
                "steps": len(report["trajectory"]),
                "final_object_state": report["final_object_state"],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def create_openai_client():
    if OPENAI_API_KEY.startswith("PASTE_"):
        raise RuntimeError("Fill OPENAI_API_KEY in the task 2 script before running task 3.")
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


def run_product_manipulation_task(
    client: Any,
    output_dir: Path,
    shelf_fbx: Path = DEFAULT_SHELF_FBX,
    instruction: str = DEFAULT_INSTRUCTION,
    width: int = 900,
    height: int = 600,
    max_steps: int = 10,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    raw_responses_dir = output_dir / "raw_responses"
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_task3_scenario(shelf_fbx)
    object_state = {
        "target_object_id": scenario["target_object"]["object_id"],
        "target_location": "source_shelf_slot",
        "held_object_id": None,
    }
    camera_state = default_initial_navigation_state(shelf_fbx)
    trajectory: list[dict[str, Any]] = []
    operation_trace: list[dict[str, Any]] = []
    question_trace: list[dict[str, Any]] = []
    done = False

    try:
        for step_idx in range(max_steps):
            input_image = images_dir / f"step_{step_idx:03d}_input.png"
            observation = render_navigation_observation(shelf_fbx, input_image, camera_state, width=width, height=height)
            _annotate_manipulation_scene(input_image, object_state, scenario)
            prompt = build_product_manipulation_prompt(
                instruction=instruction,
                scene_context=scenario,
                object_state=object_state,
                history=trajectory,
            )
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
                text={"format": _manipulation_json_schema()},
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            raw_text = _response_text(response)
            decision = _loads_json_object(raw_text)
            raw_response_path = raw_responses_dir / f"step_{step_idx:03d}_response.json"
            raw_response_path.write_text(_response_to_json(response, raw_text), encoding="utf-8")

            action = str(decision.get("action", ""))
            question_text = str(decision.get("question_text", ""))
            rationale = str(decision.get("rationale", ""))
            confidence = float(decision.get("confidence", 0.0))
            state_before = dict(object_state)
            camera_before = camera_state
            action_result = _apply_manipulation_action(action, object_state, scenario, camera_state)
            if action in NAVIGATION_ACTIONS:
                camera_state = apply_navigation_action(camera_state, action)
            if action == "AskQuestion":
                question_trace.append({"step_idx": step_idx, "question_text": question_text, "rationale": rationale})
            if action in {"PickupObject", "DropObject"}:
                operation_trace.append(
                    {
                        "step_idx": step_idx,
                        "action": action,
                        "success": action_result["success"],
                        "message": action_result["message"],
                        **{key: value for key, value in action_result.items() if key not in {"success", "message"}},
                        "state_before": state_before,
                        "state_after": dict(object_state),
                    }
                )

            step_record = {
                "step_idx": step_idx,
                "input": {
                    "model": OPENAI_MODEL,
                    "prompt": prompt,
                    "image_path": str(input_image.relative_to(output_dir)),
                    "allowed_actions": list(MANIPULATION_ACTIONS),
                },
                "observation": observation,
                "raw_response_path": str(raw_response_path.relative_to(output_dir)),
                "model_output": {
                    "action": action,
                    "question_text": question_text,
                    "rationale": rationale,
                    "confidence": confidence,
                    "raw_text": raw_text,
                },
                "executed_action": action,
                "success": action_result["success"],
                "action_message": action_result["message"],
                "state_before": state_before,
                "state_after": dict(object_state),
                "camera_state_before": camera_before.to_dict(),
                "camera_state_after": camera_state.to_dict(),
            }
            trajectory.append(step_record)
            if action == "Done" and object_state["target_location"] == "destination_shelf_slot":
                done = True
                break
    except Exception as exc:
        return _write_task3_report(
            output_dir=output_dir,
            instruction=instruction,
            scenario=scenario,
            trajectory=trajectory,
            operation_trace=operation_trace,
            question_trace=question_trace,
            object_state=object_state,
            success=False,
            termination={
                "status": "error",
                "reason": "runtime_exception",
                "done_action_required": True,
                "error_message": f"{type(exc).__name__}: {exc}",
            },
        )

    return _write_task3_report(
        output_dir=output_dir,
        instruction=instruction,
        scenario=scenario,
        trajectory=trajectory,
        operation_trace=operation_trace,
        question_trace=question_trace,
        object_state=object_state,
        success=done,
        termination={
            "status": "done" if done else "max_steps_exceeded",
            "reason": "target_object_at_destination_shelf_slot" if done else "agent_did_not_finish_relocation",
            "done_action_required": True,
        },
    )


def build_task3_scenario(shelf_fbx: Path = DEFAULT_SHELF_FBX) -> dict[str, Any]:
    product_pose = _renderer_product_pose(shelf_fbx, TARGET_RENDERER_PRODUCT_INDEX)
    product_spec = AI2THOR_COMMON_PRODUCTS[TARGET_RENDERER_PRODUCT_INDEX % len(AI2THOR_COMMON_PRODUCTS)]
    return {
        "task": "product_relocation_between_shelf_positions",
        "scene": "standalone_store_shelf_fbx",
        "shelf_fbx": str(Path(shelf_fbx)),
        "pickup_distance_threshold": PICKUP_DISTANCE_THRESHOLD,
        "target_object": {
            "object_id": f"{product_spec['object_type']}|product_index_{TARGET_RENDERER_PRODUCT_INDEX:03d}",
            "display_name_en": "red mug",
            "display_name_zh": "红色马克杯",
            "object_type": product_spec["object_type"],
            "renderer_product_index": TARGET_RENDERER_PRODUCT_INDEX,
            "renderer_shape": product_spec["shape"],
            "world_center": product_pose["center"],
            "slot": product_pose["slot"],
            "visual_description": "The actual red mug product instance rendered on the FBX shelf.",
            "source_location": "source_shelf_slot",
            "destination_location": "destination_shelf_slot",
        },
        "locations": {
            "source_shelf_slot": "the target object's initial rendered shelf slot described by the instruction",
            "destination_shelf_slot": "another shelf slot described by the instruction; it is not drawn as an overlay",
        },
        "success_condition": "target_location is destination_shelf_slot, held_object_id is null, and the final action is Done",
    }


def build_product_manipulation_prompt(
    instruction: str,
    scene_context: dict[str, Any],
    object_state: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    payload = {
        "task": "product_manipulation",
        "instruction": instruction,
        "scene_context": scene_context,
        "object_state": object_state,
        "history": history,
        "allowed_actions": list(MANIPULATION_ACTIONS),
        "action_contract": {
            "MoveAhead_Rotate_Look": "Use navigation actions to improve view of source or destination shelf positions.",
            "AskQuestion(question_text)": "Use only when the instruction or destination is ambiguous; include the question in question_text.",
            "PickupObject": (
                "The embodied agent may attempt PickupObject when it believes it is close enough. "
                "If the agent is too far from the target object's world_center, the environment returns success=false "
                "with message target_too_far_for_pickup."
            ),
            "DropObject": "Use only after a successful PickupObject when the destination shelf position is visible or close enough.",
            "Done": "Done only after object_state target_location is destination_shelf_slot and held_object_id is null.",
        },
    }
    return (
        "Generate the embodied agent's next action for a retail product manipulation task. "
        "The goal is to move the target product from one shelf position to another shelf position. "
        "Return JSON only.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _apply_manipulation_action(
    action: str,
    object_state: dict[str, Any],
    scenario: dict[str, Any],
    camera_state: Any | None = None,
) -> dict[str, Any]:
    if action not in MANIPULATION_ACTIONS:
        raise ValueError(f"Unsupported task3 action: {action!r}")
    target_id = scenario["target_object"]["object_id"]
    if action in {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "AskQuestion"}:
        return {"success": True, "message": "state_unchanged"}
    if action == "PickupObject":
        if object_state["held_object_id"] is not None:
            return {"success": False, "message": "agent_already_holding_object"}
        if object_state["target_location"] != "source_shelf_slot":
            return {"success": False, "message": "target_not_at_source_shelf_slot"}
        distance = _distance_to_target(camera_state, scenario) if camera_state is not None else math.inf
        if distance > float(scenario["pickup_distance_threshold"]):
            return {
                "success": False,
                "message": "target_too_far_for_pickup",
                "environment_feedback": "PickupObject failed: agent is too far from the target object.",
                "distance_to_target": distance,
            }
        object_state["held_object_id"] = target_id
        object_state["target_location"] = "held"
        return {"success": True, "message": "picked_up_target_object", "distance_to_target": distance}
    if action == "DropObject":
        if object_state["held_object_id"] != target_id:
            return {"success": False, "message": "target_not_held"}
        object_state["held_object_id"] = None
        object_state["target_location"] = "destination_shelf_slot"
        return {"success": True, "message": "dropped_target_at_destination_shelf_slot"}
    if action == "Done":
        success = object_state["target_location"] == "destination_shelf_slot" and object_state["held_object_id"] is None
        return {"success": success, "message": "done_success" if success else "done_before_target_relocated"}
    raise ValueError(f"Unhandled task3 action: {action!r}")


def _write_task3_report(
    *,
    output_dir: Path,
    instruction: str,
    scenario: dict[str, Any],
    trajectory: list[dict[str, Any]],
    operation_trace: list[dict[str, Any]],
    question_trace: list[dict[str, Any]],
    object_state: dict[str, Any],
    success: bool,
    termination: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "run_type": "real_gpt4o_mini_product_manipulation_task",
        "model": OPENAI_MODEL,
        "proxy": {"http": CLASH_HTTP_PROXY, "https": CLASH_HTTPS_PROXY},
        "instruction": instruction,
        "allowed_actions": list(MANIPULATION_ACTIONS),
        "scenario": scenario,
        "success": success,
        "termination": termination,
        "trajectory": trajectory,
        "operation_trace": operation_trace,
        "question_trace": question_trace,
        "final_object_state": dict(object_state),
    }
    report_path = output_dir / "task3_product_manipulation_run.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _annotate_manipulation_scene(image_path: Path, object_state: dict[str, Any], scenario: dict[str, Any]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    target_location = object_state["target_location"]
    if target_location == "destination_shelf_slot":
        destination_box = (int(w * 0.61), int(h * 0.36), int(w * 0.84), int(h * 0.49))
        _draw_target_product(draw, destination_box)
    elif target_location == "held":
        held_box = (int(w * 0.43), int(h * 0.72), int(w * 0.57), int(h * 0.91))
        draw.rectangle(held_box, outline=(255, 210, 80), width=3)
        draw.text((held_box[0] + 6, held_box[1] + 5), "HELD", fill=(255, 210, 80))
        _draw_target_product(draw, held_box)
    draw.rectangle((0, 0, w, 54), fill=(20, 24, 28))
    draw.text((12, 10), scenario["target_object"]["display_name_en"], fill=(255, 240, 210))
    draw.text((12, 30), f"target_location: {target_location}", fill=(220, 228, 235))
    image.save(image_path)


def _renderer_product_pose(shelf_fbx: Path, product_index: int) -> dict[str, Any]:
    meshes = load_store_shelf_meshes(shelf_fbx, selected_module="first")
    min_corner, max_corner = meshes.bounds()
    columns = 8
    shelf_tops = [
        float(min_corner[2] + 17.5),
        float(min_corner[2] + 41.5),
        float(min_corner[2] + 65.5),
        float(min_corner[2] + 89.5),
        float(min_corner[2] + 113.5),
    ]
    y_start = float(min_corner[1] + 12.0)
    y_end = float(max_corner[1] - 12.0)
    y_positions = [y_start + (y_end - y_start) * i / (columns - 1) for i in range(columns)]
    product_spec = AI2THOR_COMMON_PRODUCTS[product_index % len(AI2THOR_COMMON_PRODUCTS)]
    row = (product_index // columns) % len(shelf_tops)
    column = product_index % columns
    size = product_spec["size"]
    jitter = ((product_index * 17) % 9 - 4) * 0.35
    center = [
        float(min_corner[0] + 9.0 + ((product_index % 3) - 1) * 2.1),
        float(y_positions[column] + jitter),
        float(shelf_tops[row] + size[2] * 0.5 + 0.8),
    ]
    return {"center": center, "slot": {"index": product_index, "row": row, "column": column}}


def _distance_to_target(camera_state: Any, scenario: dict[str, Any]) -> float:
    target_center = scenario["target_object"]["world_center"]
    camera_position = camera_state.position
    return math.sqrt(
        (float(camera_position[0]) - float(target_center[0])) ** 2
        + (float(camera_position[1]) - float(target_center[1])) ** 2
        + (float(camera_position[2]) - float(target_center[2])) ** 2
    )


def _draw_target_product(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    pad_x = max(8, (x1 - x0) // 4)
    pad_y = max(8, (y1 - y0) // 5)
    mug = (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y)
    draw.rounded_rectangle(mug, radius=4, fill=(190, 54, 62), outline=(95, 26, 32), width=3)
    handle = (mug[2] - 4, mug[1] + 10, mug[2] + 18, mug[3] - 10)
    draw.arc(handle, start=-80, end=80, fill=(190, 54, 62), width=5)
    draw.text((mug[0] + 8, mug[1] + 12), "MUG", fill=(255, 235, 230))


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


def _manipulation_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "product_manipulation_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": list(MANIPULATION_ACTIONS)},
                "question_text": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["action", "question_text", "rationale", "confidence"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
