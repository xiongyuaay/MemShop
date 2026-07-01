from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Any, Dict, Iterable


def dist2(a: Dict[str, float], b: Dict[str, float]) -> float:
    return (a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2


def nearest_reachable_position(positions: Iterable[Dict[str, float]], target_pos: Dict[str, float]) -> Dict[str, float]:
    return min(positions, key=lambda p: dist2(p, target_pos))


def yaw_toward(agent_pos: Dict[str, float], target_pos: Dict[str, float]) -> float:
    dx = target_pos["x"] - agent_pos["x"]
    dz = target_pos["z"] - agent_pos["z"]
    return degrees(atan2(dx, dz))


def navigation_trace(positions: Iterable[Dict[str, float]], target_meta: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = target_meta["position"]
    chosen = nearest_reachable_position(positions, target_pos)
    yaw = yaw_toward(chosen, target_pos)
    return {
        "chosen_reachable_position": chosen,
        "distance_to_target": sqrt(dist2(chosen, target_pos)),
        "teleport_success": None,
        "sim_actions": [
            {"action": "GetReachablePositions"},
            {
                "action": "Teleport",
                "position": chosen,
                "rotation": {"x": 0, "y": yaw, "z": 0},
                "horizon": 30,
                "standing": True,
            },
        ],
    }


def build_open_receptacle_step(shelf_region: Dict[str, Any]) -> Dict[str, Any] | None:
    if not shelf_region.get("openable") or shelf_region.get("isOpen"):
        return None
    object_id = shelf_region.get("source_object_id") or shelf_region.get("object_id")
    return _step("open_receptacle", {"object_id": object_id}, [{"action": "OpenObject", "objectId": object_id}])


def build_search_and_look_steps(product: Dict[str, Any], query: str) -> list[Dict[str, Any]]:
    return [
        _step("search_object", {"query": query, "candidate_product_id": product.get("product_id")}, [{"action": "Done"}]),
        _step("look_at", {"object_id": product["object_id"]}, [{"action": "Teleport"}]),
    ]


def build_pick_place_steps(product: Dict[str, Any], receptacle_object_id: str, demo_mode: bool = False) -> list[Dict[str, Any]]:
    pickup_action = {"action": "PickupObject", "objectId": product["object_id"]}
    if demo_mode:
        pickup_action["forceAction"] = True
    return [
        _step("pick_object", {"object_id": product["object_id"]}, [pickup_action]),
        _step("place_object", {"target": receptacle_object_id}, [{"action": "PutObject", "objectId": receptacle_object_id}]),
    ]


def build_visibility_recovery_actions(max_attempts: int = 3) -> list[Dict[str, Any]]:
    pattern = [
        {"action": "RotateRight"},
        {"action": "RotateLeft"},
        {"action": "LookDown"},
        {"action": "LookUp"},
    ]
    return pattern[:max(0, max_attempts)]


def verify_placement(object_meta: Dict[str, Any], target_receptacle_id: str) -> bool:
    return target_receptacle_id in (object_meta.get("parentReceptacles") or [])


def _step(action: str, args: Dict[str, Any], sim_actions: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "action": action,
        "args": args,
        "preconditions": [],
        "expected_effects": [],
        "sim_actions": sim_actions,
        "success": True,
        "failure_reason": None,
    }
