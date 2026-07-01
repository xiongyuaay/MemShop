from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from retail_thor.episode_schema import validate_episode


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=str(ROOT / "outputs" / "episodes"))
    args = parser.parse_args()

    episode_dir = Path(args.episodes)
    errors = []
    checked = 0
    for path in iter_episode_files(episode_dir):
        checked += 1
        try:
            validate_episode_file(path)
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})

    report = {"checked": checked, "schema_errors": len(errors), "errors": errors}
    report_dir = ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def iter_episode_files(episode_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in episode_dir.glob("episode_*.json")
        if path.stem.startswith("episode_") and "_step_" not in path.stem
    )


def validate_episode_file(path: Path) -> None:
    episode = json.loads(path.read_text(encoding="utf-8"))
    validate_episode(episode)
    _validate_paths(path, episode)
    _validate_target_products(episode)
    _validate_shelf_traceability(episode)
    _validate_action_args(episode)


def _validate_paths(episode_path: Path, episode: dict) -> None:
    base = episode_path.parents[1]
    for obs in episode.get("observations", []):
        for field in ("rgb_path", "depth_path", "segmentation_path", "metadata_path"):
            value = obs.get(field)
            if not value:
                continue
            candidate = base / value
            if not candidate.exists():
                raise ValueError(f"missing observation path: {value}")


def _validate_target_products(episode: dict) -> None:
    catalog = episode.get("product_catalog", [])
    product_ids = {product.get("product_id") for product in catalog}
    object_ids = {product.get("object_id") for product in catalog}
    target = episode.get("target", {})

    for field, valid_ids in (
        ("target_product_id", product_ids),
        ("substitute_product_id", product_ids),
        ("target_object_id", object_ids),
        ("substitute_object_id", object_ids),
    ):
        value = target.get(field)
        if value and value not in valid_ids:
            raise ValueError(f"{field} not found in product_catalog: {value}")


def _validate_shelf_traceability(episode: dict) -> None:
    shelf_ids = {shelf.get("shelf_id") for shelf in episode.get("shelf_graph", [])}
    target = episode.get("target", {})
    target_product_ids = {
        value for value in (target.get("target_product_id"), target.get("substitute_product_id")) if value
    }

    for product in episode.get("product_catalog", []):
        if target_product_ids and product.get("product_id") not in target_product_ids:
            continue
        shelf_id = product.get("shelf_id")
        if not shelf_id or shelf_id == "unknown_shelf" or shelf_id not in shelf_ids:
            raise ValueError(f"shelf_id is not traceable for product {product.get('product_id')}: {shelf_id}")


def _validate_action_args(episode: dict) -> None:
    required_args = {
        "navigate_to_shelf": ("shelf_id",),
        "open_receptacle": ("object_id",),
        "search_object": ("query",),
        "look_at": ("object_id",),
        "ask_npc": ("instruction",),
        "recommend_substitute": ("product_id",),
        "pick_object": ("object_id",),
        "place_object": ("target",),
    }
    for idx, step in enumerate(episode.get("high_level_plan", [])):
        action = step.get("action")
        args = step.get("args", {})
        for arg in required_args.get(action, ()):
            if args.get(arg) in (None, ""):
                raise ValueError(f"high_level_plan[{idx}] missing {action}.{arg}")


if __name__ == "__main__":
    raise SystemExit(main())
