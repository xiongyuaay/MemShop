from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from PIL import Image, ImageDraw

from retail_thor.episode_schema import validate_episode
from retail_thor.generator import build_episode


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default="FloorPlan1")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "product_catalog.json"))
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-types", default="find_product,dialogue_find_or_substitute,pick_and_place")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--demo-mode", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    templates = yaml.safe_load((ROOT / "configs" / "task_templates.yaml").read_text(encoding="utf-8"))
    task_types = args.task_types.split(",")
    outputs_root = Path(args.output_root)
    out_dir = Path(args.output_dir) if args.output_dir else outputs_root / "episodes"
    episodes = generate_episodes(
        catalog=catalog,
        templates=templates,
        out_dir=out_dir,
        outputs_root=outputs_root,
        num_episodes=args.num_episodes,
        seed=args.seed,
        task_types=task_types,
        demo_mode=args.demo_mode,
        rng=rng,
    )

    outputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "episodes.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated episodes: {len(episodes)}")
    return 0


def generate_episodes(
    catalog: list[dict],
    templates: dict,
    out_dir: Path,
    outputs_root: Path,
    num_episodes: int,
    seed: int,
    task_types: list[str],
    demo_mode: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    rng = rng or random.Random(seed)
    if demo_mode and not catalog:
        catalog = demo_catalog(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for idx in range(num_episodes):
        task_type = task_types[idx % len(task_types)]
        template = rng.choice(templates[task_type])
        episode = build_episode(idx, task_type, template, catalog, seed)
        episode["provenance"]["generation"] = {
            "random_seed": seed,
            "mode": "demo_placeholder" if demo_mode else "ai2thor",
        }
        if demo_mode:
            add_demo_observations(episode, outputs_root)
        validate_episode(episode)
        path = out_dir / f"{episode['episode_id']}.json"
        path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
        episodes.append(episode)
    return episodes


def demo_catalog(seed: int = 0) -> list[dict]:
    del seed
    return [
        _demo_product("apple", "Apple|demo", "Apple", "shelf_demo_001", "苹果", "fruit", ["fresh", "healthy"], 6),
        _demo_product("bread", "Bread|demo", "Bread", "shelf_demo_002", "面包", "bakery", ["breakfast"], 8),
        _demo_product("bottle", "Bottle|demo", "Bottle", "shelf_demo_003", "饮料", "drink", ["bottled", "cold_drink"], 4),
    ]


def add_demo_observations(episode: dict, outputs_root: Path) -> None:
    image_dir = outputs_root / "images"
    metadata_dir = outputs_root / "episodes"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    sim_trace = []

    for step_idx, step in enumerate(episode["high_level_plan"]):
        image_path = image_dir / f"{episode['episode_id']}_step_{step_idx:03d}_rgb.png"
        metadata_path = metadata_dir / f"{episode['episode_id']}_step_{step_idx:03d}_metadata.json"
        _write_placeholder_image(image_path, episode, step_idx, step)
        metadata_path.write_text(
            json.dumps(
                {
                    "episode_id": episode["episode_id"],
                    "step_idx": step_idx,
                    "scene": episode["scene"],
                    "action": step["action"],
                    "demo_placeholder": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        observations.append(
            {
                "step_idx": step_idx,
                "rgb_path": str(image_path.relative_to(outputs_root)),
                "depth_path": None,
                "segmentation_path": None,
                "metadata_path": str(metadata_path.relative_to(outputs_root)),
                "agent_pose": {},
                "held_object": None,
            }
        )
        for sim_action in step.get("sim_actions", []):
            trace_item = dict(sim_action)
            trace_item["success"] = step.get("success")
            sim_trace.append(trace_item)

    episode["observations"] = observations
    episode["sim_action_trace"] = sim_trace
    episode["provenance"]["observation_source"] = "demo_placeholder"


def _demo_product(
    product_id: str,
    object_id: str,
    object_type: str,
    shelf_id: str,
    display_name_zh: str,
    category: str,
    attributes: list[str],
    price: int,
) -> dict:
    return {
        "product_id": product_id,
        "object_id": object_id,
        "object_type": object_type,
        "scene": "FloorPlan1",
        "shelf_id": shelf_id,
        "source_receptacle_id": shelf_id,
        "display_name_zh": display_name_zh,
        "category": category,
        "attributes": attributes,
        "brand": "demo",
        "price": price,
        "stock_status": "in_stock",
        "pickupable": True,
        "visible_at_start": True,
        "position": {"x": 0, "y": 1, "z": 0},
        "is_food": True,
        "scene_randomization": {
            "randomized": False,
            "random_seed": None,
            "random_spawn_success": None,
            "error_message": "",
        },
    }


def _write_placeholder_image(path: Path, episode: dict, step_idx: int, step: dict) -> None:
    image = Image.new("RGB", (800, 600), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 760, 560), outline=(90, 105, 130), width=4)
    draw.text((70, 80), f"{episode['episode_id']} step {step_idx:03d}", fill=(20, 30, 45))
    draw.text((70, 120), f"task: {episode['task_type']}", fill=(20, 30, 45))
    draw.text((70, 160), f"action: {step['action']}", fill=(20, 30, 45))
    draw.text((70, 200), "demo placeholder observation", fill=(130, 60, 20))
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
