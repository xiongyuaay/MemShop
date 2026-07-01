from __future__ import annotations

from typing import Any, Dict, Iterable


ALLOWED_ACTIONS = {
    "navigate_to_shelf",
    "open_receptacle",
    "search_object",
    "look_at",
    "ask_npc",
    "answer_user",
    "confirm_product",
    "recommend_substitute",
    "pick_object",
    "place_object",
    "compare_objects",
    "finish",
}

ALLOWED_FAILURE_REASONS = {
    None,
    "no_matching_product",
    "ambiguous_instruction_unresolved",
    "navigation_failed",
    "object_not_visible",
    "pickup_failed",
    "placement_failed",
    "schema_invalid",
    "simulator_error",
}

REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "schema_version",
    "backend",
    "scene",
    "random_seed",
    "capability_family",
    "task_type",
    "prompt",
    "target",
    "episode_idx",
    "max_steps",
    "instruction",
    "initial_state",
    "product_catalog",
    "shelf_graph",
    "npc_dialogue",
    "high_level_plan",
    "sim_action_trace",
    "observations",
    "metrics",
    "success",
    "failure_reason",
    "provenance",
}


def validate_episode(episode: Dict[str, Any]) -> None:
    missing = sorted(REQUIRED_EPISODE_FIELDS.difference(episode))
    if missing:
        raise ValueError(f"Missing required episode fields: {', '.join(missing)}")

    if not isinstance(episode["capability_family"], list) or not episode["capability_family"]:
        raise ValueError("capability_family must be a non-empty list")

    if episode["failure_reason"] not in ALLOWED_FAILURE_REASONS:
        raise ValueError(f"Unknown failure reason: {episode['failure_reason']}")

    for idx, step in enumerate(episode["high_level_plan"]):
        _validate_step(idx, step)

    if episode["success"] is False and episode["failure_reason"] is None:
        raise ValueError("failed episodes must include failure_reason")

    validate_success_conditions(episode)


def _validate_step(idx: int, step: Dict[str, Any]) -> None:
    required = {"action", "args", "preconditions", "expected_effects", "sim_actions", "success", "failure_reason"}
    missing = required.difference(step)
    if missing:
        raise ValueError(f"high_level_plan[{idx}] missing fields: {', '.join(sorted(missing))}")
    if step["action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"Unknown high-level action: {step['action']}")
    if step["failure_reason"] not in ALLOWED_FAILURE_REASONS:
        raise ValueError(f"Unknown failure reason in step {idx}: {step['failure_reason']}")
    if not isinstance(step["sim_actions"], Iterable):
        raise ValueError(f"high_level_plan[{idx}].sim_actions must be iterable")


def validate_success_conditions(episode: Dict[str, Any]) -> None:
    if not episode.get("success", False):
        return

    task_type = episode.get("task_type")
    actions = [step.get("action") for step in episode.get("high_level_plan", [])]

    if task_type == "find_product":
        _validate_find_success(episode, actions)
    elif task_type == "dialogue_find_or_substitute":
        _validate_dialogue_success(episode, actions)
    elif task_type == "pick_and_place":
        _validate_pick_and_place_success(actions, episode.get("high_level_plan", []))


def _validate_find_success(episode: Dict[str, Any], actions: list[str]) -> None:
    if "look_at" not in actions:
        raise ValueError("find_product success requires look_at")

    target = episode.get("target", {})
    expected_ids = {
        value
        for value in (
            target.get("target_product_id"),
            target.get("target_object_id"),
            target.get("substitute_product_id"),
            target.get("substitute_object_id"),
        )
        if value
    }
    if not expected_ids:
        return

    looked_ids = {
        step.get("args", {}).get("object_id")
        for step in episode.get("high_level_plan", [])
        if step.get("action") == "look_at"
    }
    if expected_ids.isdisjoint(looked_ids):
        raise ValueError("find_product success requires look_at.object_id to match target_product_id or substitute_product_id")


def _validate_dialogue_success(episode: Dict[str, Any], actions: list[str]) -> None:
    if not episode.get("npc_dialogue"):
        raise ValueError("dialogue_find_or_substitute success requires npc_dialogue")
    if "ask_npc" not in actions and "answer_user" not in actions:
        raise ValueError("dialogue_find_or_substitute success requires ask_npc or answer_user")
    target = episode.get("target", {})
    if target.get("missing_reason") and "recommend_substitute" not in actions:
        raise ValueError("dialogue_find_or_substitute missing-product success requires recommend_substitute")


def _validate_pick_and_place_success(actions: list[str], plan: list[Dict[str, Any]]) -> None:
    if "pick_object" not in actions:
        raise ValueError("pick_and_place success requires pick_object")
    if "place_object" not in actions:
        raise ValueError("pick_and_place success requires place_object")

    for action in ("pick_object", "place_object"):
        matching = [step for step in plan if step.get("action") == action]
        if not all(step.get("success") for step in matching):
            raise ValueError(f"pick_and_place success requires successful {action}")
