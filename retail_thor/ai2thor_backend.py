from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from retail_thor.config import force_ai2thor_https_downloads


class AI2ThorBackend:
    def __init__(self, config: Dict[str, Any], controller_factory=None) -> None:
        force_ai2thor_https_downloads()
        if controller_factory is None:
            from ai2thor.controller import Controller

            controller_factory = Controller

        self.config = config
        self.controller = controller_factory(**config)
        self.event = None

    def reset(self, scene: str, seed: Optional[int] = None):
        self.event = self.controller.reset(scene=scene)
        if seed is not None:
            self.step(
                {
                    "action": "InitialRandomSpawn",
                    "randomSeed": seed,
                    "forceVisible": True,
                    "numPlacementAttempts": 5,
                }
            )
        return self.event

    def step(self, action: Dict[str, Any]):
        self.event = self.controller.step(**action)
        return self.event

    def get_metadata(self) -> Dict[str, Any]:
        return self.event.metadata if self.event is not None else {}

    def save_observation(
        self,
        output_dir: Path,
        episode_id: str,
        step_idx: int,
        relative_to: Path | None = None,
    ) -> Dict[str, Any]:
        if self.event is None:
            raise RuntimeError("No AI2-THOR event available")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{episode_id}_step_{step_idx:03d}"

        rgb_path = output_dir / f"{stem}_rgb.png"
        Image.fromarray(self.event.frame).save(rgb_path)

        depth_path = None
        if getattr(self.event, "depth_frame", None) is not None:
            depth_path = output_dir / f"{stem}_depth.npy"
            np.save(depth_path, self.event.depth_frame)

        segmentation_path = None
        if getattr(self.event, "instance_segmentation_frame", None) is not None:
            segmentation_path = output_dir / f"{stem}_seg.png"
            Image.fromarray(self.event.instance_segmentation_frame).save(segmentation_path)

        metadata_path = output_dir / f"{stem}_metadata.json"
        metadata_path.write_text(json.dumps(self.event.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "step_idx": step_idx,
            "rgb_path": _format_path(rgb_path, relative_to),
            "depth_path": _format_path(depth_path, relative_to) if depth_path else None,
            "segmentation_path": _format_path(segmentation_path, relative_to) if segmentation_path else None,
            "metadata_path": _format_path(metadata_path, relative_to),
            "agent_pose": self.event.metadata.get("agent", {}),
            "held_object": self.event.metadata.get("heldObjectPose"),
            "last_action_success": self.event.metadata.get("lastActionSuccess"),
            "error_message": self.event.metadata.get("errorMessage", ""),
        }

    def stop(self) -> None:
        self.controller.stop()


def _format_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    return os.path.relpath(path, relative_to)
