import json
import sys
from pathlib import Path

import yaml


def test_build_catalog_from_inventory_dir_writes_complete_catalog(tmp_path: Path):
    from importlib import util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "03_build_catalog.py"
    sys.path.insert(0, str(script_path.parent))
    spec = util.spec_from_file_location("build_catalog_script", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    inventory_dir = tmp_path / "scene_inventory"
    inventory_dir.mkdir()
    output_path = tmp_path / "product_catalog.json"
    products_path = Path(__file__).resolve().parents[1] / "configs" / "products.yaml"
    product_rules = yaml.safe_load(products_path.read_text(encoding="utf-8"))
    inventory = {
        "scene": "FloorPlan1",
        "randomization": {"randomized": True, "random_seed": 7, "random_spawn_success": True},
        "objects": [
            {
                "objectId": "Apple|1",
                "objectType": "Apple",
                "pickupable": True,
                "visible": True,
                "position": {"x": 1, "y": 1, "z": 1},
                "parentReceptacles": ["CounterTop|1"],
            },
            {
                "objectId": "Knife|1",
                "objectType": "Knife",
                "pickupable": True,
                "visible": True,
                "parentReceptacles": ["CounterTop|1"],
            },
        ],
        "shelf_regions": [{"source_object_id": "CounterTop|1", "shelf_id": "shelf_FloorPlan1_000"}],
    }
    (inventory_dir / "FloorPlan1.json").write_text(json.dumps(inventory), encoding="utf-8")

    catalog = module.build_catalog_from_inventory_dir(inventory_dir, product_rules, output_path, seed=3)

    assert [item["object_type"] for item in catalog] == ["Apple"]
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["shelf_id"] == "shelf_FloorPlan1_000"
    assert catalog[0]["scene_randomization"]["random_seed"] == 7


def test_build_catalog_from_inventory_dir_can_filter_to_single_shelf(tmp_path: Path):
    from importlib import util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "03_build_catalog.py"
    sys.path.insert(0, str(script_path.parent))
    spec = util.spec_from_file_location("build_catalog_script_single_shelf", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    inventory_dir = tmp_path / "scene_inventory"
    inventory_dir.mkdir()
    output_path = tmp_path / "product_catalog.json"
    products_path = Path(__file__).resolve().parents[1] / "configs" / "products.yaml"
    product_rules = yaml.safe_load(products_path.read_text(encoding="utf-8"))
    inventory = {
        "scene": "FloorPlan1",
        "objects": [
            {
                "objectId": "Apple|target",
                "objectType": "Apple",
                "pickupable": True,
                "visible": True,
                "position": {"x": 1, "y": 1, "z": 1},
                "parentReceptacles": ["CounterTop|target"],
            },
            {
                "objectId": "Bread|target",
                "objectType": "Bread",
                "pickupable": True,
                "visible": True,
                "position": {"x": 2, "y": 1, "z": 1},
                "parentReceptacles": ["CounterTop|target"],
            },
            {
                "objectId": "Bottle|other",
                "objectType": "Bottle",
                "pickupable": True,
                "visible": True,
                "position": {"x": 3, "y": 1, "z": 1},
                "parentReceptacles": ["CounterTop|other"],
            },
        ],
        "shelf_regions": [
            {"source_object_id": "CounterTop|target", "shelf_id": "shelf_FloorPlan1_010"},
            {"source_object_id": "CounterTop|other", "shelf_id": "shelf_FloorPlan1_023"},
        ],
    }
    (inventory_dir / "FloorPlan1.json").write_text(json.dumps(inventory), encoding="utf-8")

    catalog = module.build_catalog_from_inventory_dir(
        inventory_dir,
        product_rules,
        output_path,
        seed=3,
        single_shelf_config={
            "enabled": True,
            "scene": "FloorPlan1",
            "shelf_id": "shelf_FloorPlan1_010",
            "source_object_id": "CounterTop|target",
        },
    )

    assert [item["object_type"] for item in catalog] == ["Apple", "Bread"]
    assert {item["shelf_id"] for item in catalog} == {"shelf_FloorPlan1_010"}
    assert json.loads(output_path.read_text(encoding="utf-8")) == catalog
