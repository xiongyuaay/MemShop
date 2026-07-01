from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from retail_thor.catalog import build_product_catalog
from retail_thor.catalog import assert_catalog_complete
from retail_thor.single_shelf import filter_catalog_to_single_shelf
from retail_thor.single_shelf import load_single_shelf_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inventory-dir", default=str(ROOT / "data" / "scene_inventory"))
    parser.add_argument("--output", default=str(ROOT / "data" / "product_catalog.json"))
    parser.add_argument("--single-shelf-config", default=str(ROOT / "configs" / "single_shelf.yaml"))
    parser.add_argument("--disable-single-shelf", action="store_true")
    args = parser.parse_args()

    rules = yaml.safe_load((ROOT / "configs" / "products.yaml").read_text(encoding="utf-8"))
    out_path = Path(args.output)
    single_shelf_config = {"enabled": False} if args.disable_single_shelf else load_single_shelf_config(args.single_shelf_config)
    all_products = build_catalog_from_inventory_dir(
        Path(args.inventory_dir),
        rules,
        out_path,
        seed=args.seed,
        single_shelf_config=single_shelf_config,
    )
    print(f"products: {len(all_products)}")
    return 0


def build_catalog_from_inventory_dir(
    inventory_dir: Path,
    product_rules: dict,
    output_path: Path,
    seed: int = 0,
    single_shelf_config: dict | None = None,
) -> list[dict]:
    all_products = []
    for inventory_path in sorted(inventory_dir.glob("*.json")):
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        shelf_lookup = {shelf["source_object_id"]: shelf["shelf_id"] for shelf in inventory.get("shelf_regions", [])}
        scene_products = build_product_catalog(
            inventory["scene"],
            inventory.get("objects", []),
            product_rules,
            shelf_lookup,
            seed=seed,
        )
        scene_randomization = inventory.get(
            "randomization",
            {"randomized": False, "random_seed": None, "random_spawn_success": None, "error_message": ""},
        )
        for product in scene_products:
            product["scene_randomization"] = scene_randomization
        all_products.extend(scene_products)

    all_products = filter_catalog_to_single_shelf(all_products, single_shelf_config)
    assert_catalog_complete(all_products)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_products, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_products


if __name__ == "__main__":
    raise SystemExit(main())
