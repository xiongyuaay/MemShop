from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_SHELF_TYPES = {"CounterTop", "Cabinet", "Fridge", "DiningTable", "TableTop", "Shelf", "Drawer"}
DEFAULT_CART_TYPES = {"Bowl", "Plate", "DiningTable", "CounterTop", "Sink"}


def normalize_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "objectId": obj.get("objectId"),
        "objectType": obj.get("objectType"),
        "name": obj.get("name"),
        "position": obj.get("position", {}),
        "rotation": obj.get("rotation", {}),
        "visible": obj.get("visible", False),
        "pickupable": obj.get("pickupable", False),
        "receptacle": obj.get("receptacle", False),
        "openable": obj.get("openable", False),
        "isOpen": obj.get("isOpen", False),
        "parentReceptacles": obj.get("parentReceptacles") or [],
        "receptacleObjectIds": obj.get("receptacleObjectIds") or [],
        "axisAlignedBoundingBox": obj.get("axisAlignedBoundingBox"),
        "objectOrientedBoundingBox": obj.get("objectOrientedBoundingBox"),
    }


def build_shelf_regions(scene: str, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    shelves = []
    for obj in objects:
        if not obj.get("receptacle") or obj.get("objectType") not in DEFAULT_SHELF_TYPES:
            continue
        shelves.append(
            {
                "shelf_id": f"shelf_{scene}_{len(shelves):03d}",
                "scene": scene,
                "source_object_id": obj["objectId"],
                "source_object_type": obj["objectType"],
                "position": obj.get("position", {}),
                "openable": obj.get("openable", False),
                "isOpen": obj.get("isOpen", False),
                "contains_object_ids": obj.get("receptacleObjectIds") or [],
                "visible_at_start": obj.get("visible", False),
                "semantic_zone": _semantic_zone(obj.get("objectType")),
            }
        )
    return shelves


def choose_cart_receptacle(scene: str, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    for obj in objects:
        if obj.get("receptacle") and obj.get("objectType") in DEFAULT_CART_TYPES:
            return {
                "cart_receptacle_id": obj["objectId"],
                "cart_semantic_id": f"cart_{scene}_000",
                "is_abstract": False,
                "position": obj.get("position", {}),
            }
    return {
        "cart_receptacle_id": None,
        "cart_semantic_id": f"cart_{scene}_abstract",
        "is_abstract": True,
        "position": {},
    }


def build_scene_inventory(
    scene: str,
    objects: List[Dict[str, Any]],
    randomization: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    shelves = build_shelf_regions(scene, objects)
    cart = choose_cart_receptacle(scene, objects)
    return {
        "scene": scene,
        "objects": objects,
        "shelf_regions": shelves,
        "cart": cart,
        "randomization": randomization
        or {
            "randomized": False,
            "random_seed": None,
            "random_spawn_success": None,
            "error_message": "",
        },
        "stats": {
            "num_objects": len(objects),
            "num_pickupable": sum(1 for obj in objects if obj.get("pickupable")),
            "num_receptacles": sum(1 for obj in objects if obj.get("receptacle")),
            "num_shelves": len(shelves),
        },
    }


def _semantic_zone(object_type: str | None) -> str:
    if object_type in {"Fridge"}:
        return "cold_shelf"
    if object_type in {"Cabinet", "Drawer"}:
        return "closed_shelf"
    return "open_shelf"
