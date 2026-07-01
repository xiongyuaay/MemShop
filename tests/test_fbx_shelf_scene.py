from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from retail_thor.fbx_shelf_scene import (
    build_fbx_shelf_scene_spec,
    load_store_shelf_meshes,
    render_fbx_store_shelf_scene,
)


ROOT = Path(__file__).resolve().parents[2]
SHELF_FBX = ROOT / "store-shelves" / "source" / "grocery_shelf.fbx"


def test_scene_spec_uses_standalone_fbx_shelf_without_room_or_statue(tmp_path):
    spec = build_fbx_shelf_scene_spec(SHELF_FBX, tmp_path, product_count=28)

    assert spec["source"] == "fbx_store_shelf_scene"
    assert spec["scene_mode"] == "standalone_asset"
    assert spec["renderer"] == "python_software_renderer"
    assert spec["shelf_asset"]["path"] == str(SHELF_FBX)
    assert spec["environment"]["room"] is None
    assert spec["environment"]["background"] == "plain studio background"
    assert spec["product_source"]["type"] == "ai2thor_common_object_categories"
    assert "Statue" in spec["excluded_object_types"]
    assert all(product["object_type"] != "Statue" for product in spec["products"])
    assert len(spec["products"]) >= 28


def test_default_scene_spec_uses_enough_products_to_fill_the_shelf(tmp_path):
    spec = build_fbx_shelf_scene_spec(SHELF_FBX, tmp_path)

    assert len(spec["products"]) >= 40


def test_fbx_loader_extracts_single_shelf_module_from_store_shelves_asset():
    meshes = load_store_shelf_meshes(SHELF_FBX, selected_module="first")

    assert len(meshes) >= 5
    assert all(mesh.triangles for mesh in meshes)
    assert {mesh.root_model_name for mesh in meshes} == {"grocery_shelf_001"}
    min_corner, max_corner = meshes.bounds()
    assert max_corner[2] - min_corner[2] > 100
    assert max_corner[2] < 150


def test_render_fbx_store_shelf_scene_writes_report_png_and_manifest(tmp_path):
    manifest = render_fbx_store_shelf_scene(
        output_dir=tmp_path,
        shelf_fbx=SHELF_FBX,
        width=640,
        height=420,
        product_count=28,
    )

    image_path = tmp_path / "fbx_store_shelf_products.png"
    manifest_path = tmp_path / "manifest.json"

    assert image_path.exists()
    assert manifest_path.exists()
    assert manifest["screenshots"][0]["path"] == image_path.name
    assert len(manifest["products"]) >= 28

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["shelf_asset"]["path"] == str(SHELF_FBX)
    assert saved_manifest["environment"]["room"] is None

    image = Image.open(image_path)
    assert image.size == (640, 420)
    assert len(image.convert("RGB").getcolors(maxcolors=1_000_000)) > 32
