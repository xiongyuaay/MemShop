from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from retail_thor.fbx_shelf_scene import load_store_shelf_meshes, render_fbx_store_shelf_image
from retail_thor.navigation_agent import NAVIGATION_ACTIONS, normalize_navigation_action


@dataclass(frozen=True)
class NavigationCameraState:
    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_initial_navigation_state(shelf_fbx: Path | str) -> NavigationCameraState:
    meshes = load_store_shelf_meshes(shelf_fbx, selected_module="first")
    min_corner, max_corner = meshes.bounds()
    center = (min_corner + max_corner) / 2.0
    position = (
        float(min_corner[0] - 155.0),
        float(min_corner[1] - 70.0),
        float(center[2] + 12.0),
    )
    yaw = math.degrees(math.atan2(float(center[1] - position[1]), float(center[0] - position[0])))
    return NavigationCameraState(position=position, yaw_degrees=yaw, pitch_degrees=-2.0)


def apply_navigation_action(
    state: NavigationCameraState,
    action: str,
    move_distance: float = 26.0,
    turn_degrees: float = 14.0,
    look_degrees: float = 8.0,
) -> NavigationCameraState:
    normalized = normalize_navigation_action(action)
    x, y, z = state.position
    if normalized == "Done":
        return state
    if normalized == "MoveAhead":
        yaw = math.radians(state.yaw_degrees)
        return NavigationCameraState(
            position=(x + math.cos(yaw) * move_distance, y + math.sin(yaw) * move_distance, z),
            yaw_degrees=state.yaw_degrees,
            pitch_degrees=state.pitch_degrees,
        )
    if normalized == "RotateLeft":
        return NavigationCameraState(state.position, state.yaw_degrees - turn_degrees, state.pitch_degrees)
    if normalized == "RotateRight":
        return NavigationCameraState(state.position, state.yaw_degrees + turn_degrees, state.pitch_degrees)
    if normalized == "LookUp":
        return NavigationCameraState(state.position, state.yaw_degrees, min(30.0, state.pitch_degrees + look_degrees))
    if normalized == "LookDown":
        return NavigationCameraState(state.position, state.yaw_degrees, max(-42.0, state.pitch_degrees - look_degrees))
    raise ValueError(f"unsupported navigation action: {action}")


def render_navigation_observation(
    shelf_fbx: Path | str,
    image_path: Path | str,
    state: NavigationCameraState,
    width: int = 900,
    height: int = 600,
    product_count: int = 40,
) -> dict[str, Any]:
    eye = np.asarray(state.position, dtype=float)
    target = eye + _forward_vector(state) * 140.0
    image = render_fbx_store_shelf_image(
        shelf_fbx=shelf_fbx,
        width=width,
        height=height,
        product_count=product_count,
        selected_module="first",
        camera_eye=eye,
        camera_target=target,
    )
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "rgb_path": str(path),
        "width": width,
        "height": height,
        "camera_state": state.to_dict(),
    }


def build_navigation_prompt(
    instruction: str,
    navigation_history: Sequence[dict[str, Any]] | None = None,
    scene_context: dict[str, Any] | None = None,
) -> str:
    payload = {
        "instruction": instruction,
        "allowed_actions": list(NAVIGATION_ACTIONS),
        "navigation_history": list(navigation_history or []),
        "scene_context": scene_context or {},
        "termination_policy": {
            "done_when": (
                "At least one matching target instance is clearly visible and identifiable, "
                "and it is near the image center or large enough in the image for stable confirmation."
            ),
            "target_match": "For this task, any one blue, thin, cylindrical product instance is sufficient.",
            "not_required": "Do not require pickup, perfect alignment, or interaction distance for this find-only task.",
            "continue_when": (
                "If the target is only peripheral, too small, heavily occluded, or ambiguous with a blue box/book, "
                "continue with MoveAhead, RotateLeft/RotateRight, LookUp, or LookDown."
            ),
        },
    }
    return (
        "You are the navigation brain for an embodied agent in a standalone retail shelf scene. "
        "Choose exactly one next navigation action from allowed_actions. "
        "Use MoveAhead to approach visible targets, RotateLeft/RotateRight to turn, "
        "LookUp/LookDown to adjust camera pitch. "
        "Use Done only under the termination_policy in the JSON payload: the requested object must be clearly found, "
        "visually confirmable, and sufficiently centered or close. "
        "Return JSON only.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _forward_vector(state: NavigationCameraState) -> np.ndarray:
    yaw = math.radians(state.yaw_degrees)
    pitch = math.radians(state.pitch_degrees)
    return np.asarray(
        [
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=float,
    )
