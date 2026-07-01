import json
from pathlib import Path

from PIL import Image

from retail_thor.visual_scene import generate_retail_shelf_screenshots


def test_generate_retail_shelf_screenshots_writes_three_nonblank_pngs_and_manifest(tmp_path: Path):
    catalog = [
        {
            "product_id": "apple",
            "display_name_en": "apple",
            "category": "fruit",
            "brand": "daily fresh",
            "price": 6,
            "barcode": "690000000001",
            "expiration_date": "2026-07-10",
        },
        {
            "product_id": "bread",
            "display_name_en": "bread",
            "category": "bakery",
            "brand": "morning bake",
            "price": 8,
            "barcode": "690000000002",
            "expiration_date": "2026-07-03",
        },
    ]

    manifest = generate_retail_shelf_screenshots(tmp_path, catalog=catalog, width=640, height=480)

    assert manifest["source"] == "synthetic_report_scene"
    assert manifest["visual_style"] == "sari_sandbox_like_perspective"
    assert len(manifest["screenshots"]) == 3
    assert (tmp_path / "manifest.json").exists()

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["source"] == "synthetic_report_scene"
    assert saved_manifest["visual_style"] == "sari_sandbox_like_perspective"

    for screenshot in manifest["screenshots"]:
        image_path = tmp_path / screenshot["path"]
        assert image_path.exists()
        assert image_path.suffix == ".png"
        with Image.open(image_path) as image:
            assert image.size == (640, 480)
            assert len(image.getcolors(maxcolors=256 * 256)) > 50


def test_generate_retail_shelf_screenshots_has_stable_scene_names(tmp_path: Path):
    manifest = generate_retail_shelf_screenshots(tmp_path, catalog=[], width=320, height=240)

    assert [item["scene_id"] for item in manifest["screenshots"]] == [
        "sari_store_front",
        "sari_store_aisle",
        "sari_checkout_view",
    ]


def test_generate_retail_shelf_screenshots_draws_ceiling_and_perspective_floor(tmp_path: Path):
    generate_retail_shelf_screenshots(tmp_path, catalog=[], width=640, height=480)

    with Image.open(tmp_path / "sari_store_front.png") as image:
        pixels = image.convert("RGB")
        top_region = pixels.crop((0, 0, 640, 160))
        bottom_region = pixels.crop((0, 330, 640, 480))
        top_data = top_region.load()
        bottom_data = bottom_region.load()

        bright_ceiling_pixels = sum(
            1
            for x in range(top_region.width)
            for y in range(top_region.height)
            if all(channel > 220 for channel in top_data[x, y])
        )
        dark_floor_pixels = sum(
            1
            for x in range(bottom_region.width)
            for y in range(bottom_region.height)
            if bottom_data[x, y][0] < 55 and bottom_data[x, y][1] < 55 and bottom_data[x, y][2] < 65
        )

    assert bright_ceiling_pixels > 1000
    assert dark_floor_pixels > 20000
