from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from PIL import Image

from retail_thor.config import controller_config_from_scene_config, force_ai2thor_https_downloads
from retail_thor.single_shelf import load_single_shelf_config


ROOT = Path(__file__).resolve().parents[1]
AI2THOR_NATIVE_STANDINS = {
    "Apple",
    "Book",
    "Bottle",
    "Bowl",
    "Bread",
    "Cup",
    "Egg",
    "Mug",
    "Plate",
    "Potato",
    "SaltShaker",
    "SoapBottle",
    "Spoon",
    "Tomato",
    "Vase",
}
EXCLUDED_SHELF_OBJECT_TYPES = {"Statue"}
DEFAULT_REAL_SHELF = {
    "enabled": True,
    "scene": "FloorPlan1",
    "shelf_id": "ai2thor_shelf_FloorPlan1_000",
    "source_object_type": "Shelf",
    "source_object_ids": [
        "Shelf|+01.75|+00.17|-02.56",
        "Shelf|+01.75|+00.55|-02.56",
        "Shelf|+01.75|+00.88|-02.56",
    ],
    "focus_object_id": "Shelf|+01.75|+00.55|-02.56",
    "verified_probe": {
        "position": {"x": 1.75, "y": 0.900999128818512, "z": -1.25},
        "rotation": {"x": 0, "y": 179.9557435, "z": 0},
        "horizon": 10,
        "standing": True,
    },
    "randomization": {
        "action": "InitialRandomSpawn",
        "randomSeed": 86,
        "forceVisible": True,
        "numPlacementAttempts": 5,
    },
    "excluded_object_types": sorted(EXCLUDED_SHELF_OBJECT_TYPES),
    "semantic_zone": "single_retail_shelf",
    "description": "AI2-THOR native Shelf objects used as one bookcase-style retail shelf.",
}


def render_real_single_shelf(
    output_dir: Path | str,
    controller_factory: Callable[..., Any] | None = None,
    controller_config: Dict[str, Any] | None = None,
    scene: str | None = None,
    single_shelf_config_path: Path | str | None = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shelf_config = _load_real_shelf_config(single_shelf_config_path)
    scene_name = scene or shelf_config.get("scene", "FloorPlan1")
    config = dict(controller_config or _load_default_controller_config())

    if controller_factory is None:
        from ai2thor.controller import Controller

        force_ai2thor_https_downloads()
        controller_factory = Controller

    controller = controller_factory(scene=scene_name, **config)
    try:
        event = controller.step(action="Done")
        randomization = shelf_config.get("randomization")
        if randomization is not None:
            event = controller.step(**randomization)

        objects = event.metadata.get("objects", [])
        focus_object_id = shelf_config.get("focus_object_id") or shelf_config["source_object_ids"][0]
        shelf = _find_object(objects, focus_object_id)
        if shelf is None:
            raise RuntimeError(f"AI2-THOR shelf object not found: {focus_object_id}")

        source_object_ids = list(shelf_config["source_object_ids"])
        final_event = _teleport_to_best_single_shelf_view(
            controller, shelf, source_object_ids, shelf_config.get("verified_probe")
        )

        final_objects = final_event.metadata.get("objects", [])
        shelf_products = _products_on_shelf(final_objects, source_object_ids)

        raw_image_path = output_path / "real_single_shelf_raw.png"
        image_path = output_path / "real_single_shelf.png"
        raw_image = Image.fromarray(final_event.frame)
        raw_image.save(raw_image_path)
        _crop_single_shelf_focus(raw_image).save(image_path)

        metadata = {
            "scene": scene_name,
            "shelf": shelf_config,
            "shelf_randomization": randomization,
            "excluded_object_types": sorted(EXCLUDED_SHELF_OBJECT_TYPES),
            "view_policy": "cropped_single_shelf_focus",
            "agent": final_event.metadata.get("agent", {}),
            "lastActionSuccess": final_event.metadata.get("lastActionSuccess"),
            "errorMessage": final_event.metadata.get("errorMessage", ""),
            "objects_on_shelf": shelf_products,
        }
        (output_path / "real_single_shelf_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        rendered_types = sorted({product["objectType"] for product in shelf_products if product.get("visible")})
        manifest = {
            "source": "ai2thor_real_render",
            "scene": scene_name,
            "shelf": shelf_config,
            "object_policy": "AI2-THOR native objects only",
            "shelf_randomization": randomization,
            "excluded_object_types": sorted(EXCLUDED_SHELF_OBJECT_TYPES),
            "view_policy": "cropped_single_shelf_focus",
            "allowed_object_types": sorted(AI2THOR_NATIVE_STANDINS),
            "rendered_object_types": rendered_types,
            "screenshots": [
                {
                    "scene_id": "real_single_shelf",
                    "path": "real_single_shelf.png",
                    "caption": "Real AI2-THOR render of one receptacle used as a retail shelf.",
                }
            ],
            "raw_screenshot_path": "real_single_shelf_raw.png",
            "metadata_path": "real_single_shelf_metadata.json",
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
    finally:
        controller.stop()


def _load_real_shelf_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    if config_path is None:
        return json.loads(json.dumps(DEFAULT_REAL_SHELF))

    config = load_single_shelf_config(config_path)
    if "source_object_ids" not in config:
        config["source_object_ids"] = [config["source_object_id"]]
    config.setdefault("focus_object_id", config["source_object_ids"][0])
    return config


def _load_default_controller_config() -> Dict[str, Any]:
    import yaml

    scene_cfg = yaml.safe_load((ROOT / "configs" / "scenes.yaml").read_text(encoding="utf-8"))
    return controller_config_from_scene_config(scene_cfg["ai2thor"])


def _find_object(objects: Iterable[Dict[str, Any]], object_id: str) -> Dict[str, Any] | None:
    return next((obj for obj in objects if obj.get("objectId") == object_id), None)


def _products_on_shelf(objects: Iterable[Dict[str, Any]], shelf_object_ids: Iterable[str]) -> List[Dict[str, Any]]:
    shelf_object_id_set = set(shelf_object_ids)
    products = []
    for obj in objects:
        if obj.get("objectType") not in AI2THOR_NATIVE_STANDINS:
            continue
        if obj.get("objectType") in EXCLUDED_SHELF_OBJECT_TYPES:
            continue
        if shelf_object_id_set.isdisjoint(obj.get("parentReceptacles") or []):
            continue
        products.append(
            {
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "visible": obj.get("visible", False),
                "pickupable": obj.get("pickupable", False),
                "position": obj.get("position", {}),
                "parentReceptacles": obj.get("parentReceptacles") or [],
            }
        )
    return products


def _teleport_to_best_single_shelf_view(
    controller: Any,
    shelf: Dict[str, Any],
    shelf_object_ids: Iterable[str],
    verified_probe: Dict[str, Any] | None = None,
) -> Any:
    shelf_object_id_list = list(shelf_object_ids)

    if verified_probe is not None:
        event = controller.step(action="Teleport", **verified_probe)
        products = _products_on_shelf(event.metadata.get("objects", []), shelf_object_id_list)
        if event.metadata.get("lastActionSuccess") and any(product.get("visible") for product in products):
            return event

    center = _object_center(shelf)
    candidates = [
        ("x_minus", _grid_position(center["x"] - 1.25, center["z"]), 90.0),
        ("x_plus", _grid_position(center["x"] + 1.25, center["z"]), -90.0),
        ("z_minus", _grid_position(center["x"], center["z"] - 1.25), 0.0),
        ("z_plus", _grid_position(center["x"], center["z"] + 1.25), 180.0),
    ]

    best_event = None
    best_score = -1
    for _, position, fallback_yaw in candidates:
        event = controller.step(
            action="Teleport",
            position=position,
            rotation={"x": 0, "y": _yaw_toward(position, center, fallback_yaw), "z": 0},
            horizon=10,
            standing=True,
        )
        products = _products_on_shelf(event.metadata.get("objects", []), shelf_object_id_list)
        score = sum(1 for product in products if product.get("visible"))
        if event.metadata.get("lastActionSuccess") and score > best_score:
            best_event = event
            best_score = score

    if best_event is None:
        return controller.step(
            action="Teleport",
            position=candidates[0][1],
            rotation={"x": 0, "y": candidates[0][2], "z": 0},
            horizon=10,
            standing=True,
        )
    return best_event


def _grid_position(x: float, z: float) -> Dict[str, float]:
    return {"x": _snap_to_grid(x), "y": 0.900999128818512, "z": _snap_to_grid(z)}


def _snap_to_grid(value: float, step: float = 0.25) -> float:
    return round(value / step) * step


def _crop_single_shelf_focus(image: Image.Image) -> Image.Image:
    width, height = image.size
    left = int(width * 0.16)
    top = 0
    right = int(width * 0.64)
    bottom = height
    if right - left < 2 or bottom - top < 2:
        return image.copy()

    cropped = image.crop((left, top, right, bottom))
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return cropped.resize((width, height), resample)


def _object_center(obj: Dict[str, Any]) -> Dict[str, float]:
    bbox = obj.get("axisAlignedBoundingBox") or {}
    center = bbox.get("center") or obj.get("position") or {}
    return {
        "x": float(center.get("x", 0.0)),
        "y": float(center.get("y", 0.0)),
        "z": float(center.get("z", 0.0)),
    }


def _yaw_toward(position: Dict[str, float], target: Dict[str, float], fallback: float) -> float:
    from math import atan2, degrees

    dx = target["x"] - position["x"]
    dz = target["z"] - position["z"]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return fallback
    return degrees(atan2(dx, dz))
