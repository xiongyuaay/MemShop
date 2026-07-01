from __future__ import annotations

from typing import Any, Dict


def summarize_episode_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    plan = episode.get("high_level_plan", [])
    actions = [step.get("action") for step in plan]
    sim_steps = sum(len(step.get("sim_actions", [])) for step in plan)

    return {
        "target_found": "look_at" in actions or episode.get("success", False),
        "dialogue_resolved": bool(episode.get("npc_dialogue")) if "ask_npc" in actions else False,
        "substitute_valid": "recommend_substitute" in actions,
        "pickup_success": _action_success(plan, "pick_object"),
        "placement_success": _action_success(plan, "place_object"),
        "num_high_level_steps": len(plan),
        "num_sim_steps": sim_steps,
        "num_dialogue_turns": len(episode.get("npc_dialogue", [])),
        "used_force_action": any(
            sim.get("forceAction") is True for step in plan for sim in step.get("sim_actions", [])
        ),
        "final_distance_to_target": episode.get("final_distance_to_target"),
    }


def _action_success(plan: list[dict[str, Any]], action: str) -> bool:
    matches = [step for step in plan if step.get("action") == action]
    return bool(matches) and all(step.get("success") for step in matches)


def check_acceptance_criteria(episodes: list[Dict[str, Any]], criteria: Dict[str, Any]) -> Dict[str, Any]:
    task_types = {episode.get("task_type") for episode in episodes}
    successful = [episode for episode in episodes if episode.get("success")]
    substitute_samples = [
        episode
        for episode in episodes
        if any(step.get("action") == "recommend_substitute" for step in episode.get("high_level_plan", []))
    ]
    pick_and_place_success = [
        episode
        for episode in episodes
        if episode.get("task_type") == "pick_and_place"
        and episode.get("success")
        and (
            episode.get("metrics", {}).get("placement_success")
            or any(step.get("action") == "place_object" and step.get("success") for step in episode.get("high_level_plan", []))
        )
    ]

    counts = {
        "episodes": len(episodes),
        "success": len(successful),
        "substitute_samples": len(substitute_samples),
        "pick_and_place_success": len(pick_and_place_success),
        "task_types": sorted(task_type for task_type in task_types if task_type),
    }
    failed_checks = []

    if counts["episodes"] < criteria.get("min_episodes", 0):
        failed_checks.append("min_episodes")
    if counts["success"] < criteria.get("min_success", 0):
        failed_checks.append("min_success")
    if counts["substitute_samples"] < criteria.get("min_substitute_samples", 0):
        failed_checks.append("min_substitute_samples")
    if counts["pick_and_place_success"] < criteria.get("min_pick_and_place_success", 0):
        failed_checks.append("min_pick_and_place_success")

    required_task_types = set(criteria.get("required_task_types", []))
    if not required_task_types.issubset(task_types):
        failed_checks.append("required_task_types")

    return {
        "passed": not failed_checks,
        "counts": counts,
        "failed_checks": failed_checks,
    }
