from __future__ import annotations

import json
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from retail_thor.ai2thor_backend import AI2ThorBackend
from retail_thor.config import controller_config_from_scene_config
from retail_thor.scene_scan import build_scene_inventory, normalize_object


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    scene_cfg = yaml.safe_load((ROOT / "configs" / "scenes.yaml").read_text(encoding="utf-8"))
    ai2thor_cfg = controller_config_from_scene_config(scene_cfg["ai2thor"])
    scenes = parse_scenes(args.scenes, scene_cfg["demo_scenes"])
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "scene_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene_idx, scene in enumerate(scenes):
        backend = AI2ThorBackend(ai2thor_cfg)
        try:
            scene_seed = args.seed + scene_idx if args.randomize else None
            spawn_event = backend.reset(scene, seed=scene_seed)
            randomization = {
                "randomized": args.randomize,
                "random_seed": scene_seed,
                "random_spawn_success": spawn_event.metadata.get("lastActionSuccess") if args.randomize else None,
                "error_message": spawn_event.metadata.get("errorMessage", "") if args.randomize else "",
            }
            event = backend.step({"action": "Done"})
            objects = [normalize_object(obj) for obj in event.metadata.get("objects", [])]
            inventory = build_scene_inventory(scene, objects, randomization=randomization)
            (out_dir / f"{scene}.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(inventory["stats"], ensure_ascii=False))
        finally:
            backend.stop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan AI2-THOR scenes into scene inventory JSON files.")
    parser.add_argument("--scenes", default=None, help="Comma-separated scenes. Defaults to configs/scenes.yaml demo_scenes.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--randomize", action="store_true", help="Run InitialRandomSpawn before scanning each scene.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for randomization; scene index is added per scene.")
    return parser.parse_args()


def parse_scenes(value: str | None, default_scenes: list[str]) -> list[str]:
    if not value:
        return list(default_scenes)
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
