from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from retail_thor.catalog import match_product
from retail_thor.metrics import summarize_episode_metrics
from retail_thor.npc import resolve_dialogue_task


def select_candidates(catalog: List[Dict[str, Any]], constraint: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [product for product in catalog if match_product(product, constraint)]


def build_find_plan(product: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
    return [
        _step("navigate_to_shelf", {"shelf_id": product.get("shelf_id", "unknown_shelf")}, ["Teleport"]),
        _step("search_object", {"query": instruction}, ["Done"]),
        _step("look_at", {"object_id": product["object_id"]}, ["Teleport"]),
        _step("answer_user", {"text": f"我找到了{product['display_name_zh']}。"}, ["Done"]),
        _step("finish", {}, ["Done"]),
    ]


def build_episode(idx: int, task_type: str, template: dict, catalog: list[dict], seed: int) -> dict:
    instruction = template["instruction"]
    constraint = template.get("constraint", {})
    candidates = select_candidates(catalog, constraint) if constraint else list(catalog)
    target = candidates[idx % len(candidates)] if candidates else None
    target_summary = {
        "constraint": constraint,
        "candidate_product_ids": [product["product_id"] for product in candidates],
    }
    success = target is not None
    failure_reason = None if success else "no_matching_product"
    dialogue = []
    plan = []
    capability = ["object_loco_navigation"]

    if task_type == "dialogue_find_or_substitute":
        capability = ["social_loco_navigation"]
        dialogue_result = resolve_dialogue_task(instruction, catalog)
        dialogue = dialogue_result["dialogue"]
        selected_product_id = dialogue_result.get("selected_product_id")
        target = next((p for p in catalog if p["product_id"] == selected_product_id), target)
        success = target is not None and not dialogue_result.get("requires_clarification", False)
        failure_reason = None if success else "ambiguous_instruction_unresolved"
        target_summary["candidate_product_ids"] = [selected_product_id] if selected_product_id else []
        substitute_turn = next((turn for turn in dialogue if turn.get("intent") == "recommend_substitute"), None)
        if substitute_turn:
            state_delta = substitute_turn.get("state_delta", {})
            target_summary.update(
                {
                    "requested_product": state_delta.get("requested_product"),
                    "missing_reason": state_delta.get("missing_reason", "not_in_scene"),
                    "substitute_product_id": selected_product_id,
                    "substitute_reason": state_delta.get("substitute_reason"),
                }
            )
    elif task_type == "pick_and_place":
        capability = ["loco_manipulation"]
        target_summary["placement"] = template.get("placement", "cart")

    if target:
        target_summary.setdefault("target_product_id", target["product_id"])
        target_summary.setdefault("target_object_id", target["object_id"])
        if "substitute_product_id" in target_summary:
            target_summary["substitute_object_id"] = target["object_id"]

        plan = build_find_plan(target, instruction)
        if task_type == "dialogue_find_or_substitute" and dialogue:
            plan.insert(0, _step("ask_npc", {"instruction": instruction}, ["Done"]))
            if any(turn["intent"] == "recommend_substitute" for turn in dialogue):
                plan.insert(1, _step("recommend_substitute", {"product_id": target["product_id"]}, ["Done"]))
        if task_type == "pick_and_place":
            plan.insert(-1, _step("pick_object", {"object_id": target["object_id"]}, ["PickupObject"]))
            plan.insert(-1, _step("place_object", {"target": template.get("placement", "cart")}, ["PutObject"]))

    scene_randomization = (
        target.get("scene_randomization")
        if target
        else {"randomized": False, "random_seed": None, "random_spawn_success": None, "error_message": ""}
    )

    episode = {
        "episode_id": f"episode_{idx:06d}",
        "schema_version": "0.1.0",
        "backend": "ai2thor",
        "scene": target.get("scene", "FloorPlan1") if target else "FloorPlan1",
        "random_seed": seed,
        "capability_family": capability,
        "task_type": task_type,
        "prompt": instruction,
        "target": target_summary,
        "episode_idx": idx,
        "max_steps": 20,
        "instruction": instruction,
        "initial_state": {"agent_pose": {}, "held_object": None},
        "product_catalog": catalog,
        "shelf_graph": build_shelf_graph(catalog),
        "npc_dialogue": dialogue,
        "high_level_plan": plan or [_step("finish", {}, ["Done"], success=False, failure_reason=failure_reason)],
        "sim_action_trace": [],
        "observations": [],
        "metrics": {},
        "success": success,
        "failure_reason": failure_reason,
        "provenance": {
            "generator": "retail_thor.generator.build_episode",
            "generated_at": datetime.now().isoformat(),
            "randomization": scene_randomization,
        },
    }
    episode["metrics"] = summarize_episode_metrics(episode)
    return episode


def build_shelf_graph(catalog: list[dict]) -> list[dict]:
    seen = set()
    shelves = []
    for product in catalog:
        shelf_id = product.get("shelf_id")
        if not shelf_id or shelf_id == "unknown_shelf" or shelf_id in seen:
            continue
        seen.add(shelf_id)
        shelves.append(
            {
                "shelf_id": shelf_id,
                "source_object_id": product.get("source_receptacle_id", shelf_id),
            }
        )
    return shelves


def _step(
    action: str,
    args: Dict[str, Any],
    sim_actions: List[str],
    success: bool = True,
    failure_reason: str | None = None,
) -> Dict[str, Any]:
    return {
        "action": action,
        "args": args,
        "preconditions": [],
        "expected_effects": [],
        "sim_actions": [{"action": sim_action} for sim_action in sim_actions],
        "success": success,
        "failure_reason": failure_reason,
    }
