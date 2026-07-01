from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


DEFAULT_PRODUCTS = [
    {"display_name_en": "Apple", "category": "fruit", "brand": "Daily Fresh", "price": 6, "barcode": "690000000001"},
    {"display_name_en": "Bread", "category": "bakery", "brand": "Morning Bake", "price": 8, "barcode": "690000000002"},
    {"display_name_en": "Drink", "category": "drink", "brand": "Cool Spring", "price": 4, "barcode": "690000000003"},
    {"display_name_en": "Tomato", "category": "vegetable", "brand": "Green Farm", "price": 5, "barcode": "690000000004"},
    {"display_name_en": "Milk", "category": "dairy", "brand": "North Dairy", "price": 9, "barcode": "690000000005"},
    {"display_name_en": "Cereal", "category": "breakfast", "brand": "Bright Day", "price": 12, "barcode": "690000000006"},
    {"display_name_en": "Noodles", "category": "staple", "brand": "Quick Bowl", "price": 7, "barcode": "690000000007"},
    {"display_name_en": "Juice", "category": "drink", "brand": "Sun Orchard", "price": 6, "barcode": "690000000008"},
]

CATEGORY_COLORS = {
    "fruit": (211, 62, 67),
    "vegetable": (46, 139, 87),
    "bakery": (188, 128, 61),
    "drink": (44, 124, 196),
    "dairy": (238, 238, 232),
    "breakfast": (240, 178, 50),
    "staple": (185, 91, 48),
    "container": (122, 132, 143),
}


def generate_retail_shelf_screenshots(
    output_dir: Path | str,
    catalog: Iterable[Dict[str, Any]] | None = None,
    width: int = 800,
    height: int = 600,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    products = _normalize_products(catalog)

    scene_specs: list[tuple[str, str, str, Callable[[ImageDraw.ImageDraw, list[dict], int, int], None]]] = [
        ("sari_store_front", "sari_store_front.png", "SARI-like first-person retail store view", _draw_sari_store_front),
        ("sari_store_aisle", "sari_store_aisle.png", "SARI-like aisle view with shelves", _draw_sari_store_aisle),
        ("sari_checkout_view", "sari_checkout_view.png", "SARI-like checkout view with scanner", _draw_sari_checkout_view),
    ]

    screenshots = []
    for scene_id, filename, caption, draw_scene in scene_specs:
        image = _base_canvas(width, height)
        draw = ImageDraw.Draw(image)
        draw_scene(draw, products, width, height)
        image.save(output_path / filename)
        screenshots.append({"scene_id": scene_id, "path": filename, "caption": caption})

    manifest = {
        "source": "synthetic_report_scene",
        "visual_style": "sari_sandbox_like_perspective",
        "note": "Report-oriented SARI-like synthetic perspective scenes; these are not simulator-rendered frames.",
        "width": width,
        "height": height,
        "screenshots": screenshots,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_catalog(path: Path | str | None) -> list[dict]:
    if path is None:
        return []
    catalog_path = Path(path)
    if not catalog_path.exists():
        return []
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def _normalize_products(catalog: Iterable[Dict[str, Any]] | None) -> list[dict]:
    products = []
    for item in catalog or []:
        name = item.get("display_name_en") or item.get("object_type") or item.get("product_id") or "Product"
        products.append(
            {
                "name": str(name).title()[:14],
                "category": str(item.get("category", "staple")),
                "brand": str(item.get("brand", "Store Brand")).title()[:16],
                "price": item.get("price", 5),
                "barcode": str(item.get("barcode", item.get("product_id", "000000000000"))),
            }
        )
    if products:
        return products[:24]
    return [
        {
            "name": item["display_name_en"],
            "category": item["category"],
            "brand": item["brand"],
            "price": item["price"],
            "barcode": item["barcode"],
        }
        for item in DEFAULT_PRODUCTS
    ]


def _base_canvas(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (238, 241, 239))
    pixels = image.load()
    for y in range(height):
        shade = int(238 - (y / max(1, height)) * 34)
        for x in range(width):
            pixels[x, y] = (shade, min(244, shade + 5), min(242, shade + 3))
    return image


def _draw_sari_store_front(draw: ImageDraw.ImageDraw, products: list[dict], width: int, height: int) -> None:
    font = _font(max(10, width // 76))
    small_font = _font(max(8, width // 95))
    room = _draw_low_poly_store_room(draw, width, height)

    _draw_sari_shelf(draw, int(width * 0.38), int(height * 0.47), int(width * 0.16), int(height * 0.17), "Shelf 4", products, font, small_font)
    _draw_sari_shelf(draw, int(width * 0.58), int(height * 0.46), int(width * 0.19), int(height * 0.19), "Shelf 8", products[2:] + products[:2], font, small_font)
    _draw_sari_shelf(draw, int(width * 0.78), int(height * 0.48), int(width * 0.13), int(height * 0.2), "Shelf 3", products[4:] + products[:4], font, small_font)

    counter_y = int(height * 0.62)
    draw.polygon(
        [
            (int(width * -0.02), counter_y),
            (width, int(height * 0.58)),
            (width, height),
            (0, height),
        ],
        fill=(28, 25, 29),
    )
    draw.line((0, counter_y, width, int(height * 0.58)), fill=(236, 238, 235), width=max(2, width // 220))
    draw.polygon(
        [
            (0, int(height * 0.43)),
            (int(width * 0.18), int(height * 0.49)),
            (int(width * 0.18), int(height * 0.73)),
            (0, int(height * 0.77)),
        ],
        fill=(112, 122, 132),
        outline=(226, 230, 228),
    )
    draw.polygon(
        [
            (0, int(height * 0.42)),
            (int(width * 0.18), int(height * 0.49)),
            (int(width * 0.24), int(height * 0.47)),
            (int(width * 0.08), int(height * 0.38)),
        ],
        fill=(204, 212, 216),
        outline=(238, 240, 238),
    )
    _draw_report_badge(draw, width, height, small_font)


def _draw_sari_store_aisle(draw: ImageDraw.ImageDraw, products: list[dict], width: int, height: int) -> None:
    font = _font(max(10, width // 76))
    small_font = _font(max(8, width // 95))
    room = _draw_low_poly_store_room(draw, width, height, back_wall=(143, 164, 148), right_wall=(180, 194, 144))

    for idx, y_factor in enumerate((0.42, 0.52, 0.64)):
        _draw_sari_shelf(draw, int(width * 0.23), int(height * y_factor), int(width * (0.25 + idx * 0.06)), int(height * 0.18), f"Shelf L{idx + 1}", products[idx:] + products[:idx], font, small_font)
        _draw_sari_shelf(draw, int(width * 0.77), int(height * y_factor), int(width * (0.25 + idx * 0.06)), int(height * 0.18), f"Shelf R{idx + 1}", products[-idx:] + products[:-idx], font, small_font)

    lane = [
        (int(width * 0.43), room["floor_y"]),
        (int(width * 0.57), room["floor_y"]),
        (int(width * 0.73), height),
        (int(width * 0.27), height),
    ]
    draw.polygon(lane, fill=(45, 42, 48), outline=(76, 73, 78))
    for row in range(5):
        y = int(room["floor_y"] + row * (height - room["floor_y"]) / 5)
        draw.line((int(width * 0.48), y, int(width * 0.52), y), fill=(88, 84, 90), width=1)
    _draw_report_badge(draw, width, height, small_font)


def _draw_sari_checkout_view(draw: ImageDraw.ImageDraw, products: list[dict], width: int, height: int) -> None:
    font = _font(max(10, width // 76))
    small_font = _font(max(8, width // 95))
    _draw_low_poly_store_room(draw, width, height, back_wall=(146, 165, 150), right_wall=(187, 198, 154))

    _draw_sari_shelf(draw, int(width * 0.34), int(height * 0.46), int(width * 0.17), int(height * 0.17), "Shelf 2", products, font, small_font)
    _draw_sari_shelf(draw, int(width * 0.66), int(height * 0.46), int(width * 0.18), int(height * 0.18), "Shelf 5", products[3:] + products[:3], font, small_font)

    counter = [
        (int(width * 0.05), int(height * 0.58)),
        (int(width * 0.95), int(height * 0.55)),
        (width, height),
        (0, height),
    ]
    draw.polygon(counter, fill=(33, 30, 34), outline=(230, 233, 230))
    draw.line((int(width * 0.05), int(height * 0.58), int(width * 0.95), int(height * 0.55)), fill=(235, 238, 234), width=max(2, width // 220))

    scanner = (int(width * 0.47), int(height * 0.61), int(width * 0.59), int(height * 0.69))
    draw.rounded_rectangle(scanner, radius=6, fill=(47, 58, 64), outline=(13, 19, 23), width=2)
    draw.line((scanner[0] + 8, scanner[1] + 17, scanner[2] - 8, scanner[3] - 13), fill=(95, 235, 165), width=3)
    draw.text((scanner[0] + 6, scanner[3] + 6), "scanner", fill=(218, 224, 218), font=small_font)

    basket = (int(width * 0.12), int(height * 0.62), int(width * 0.34), int(height * 0.82))
    draw.rectangle(basket, fill=(154, 60, 54), outline=(224, 120, 105), width=2)
    for x in range(basket[0] + 12, basket[2], max(12, width // 42)):
        draw.line((x, basket[1] + 7, x, basket[3] - 7), fill=(235, 150, 135), width=1)
    for y in range(basket[1] + 14, basket[3], max(12, height // 42)):
        draw.line((basket[0] + 7, y, basket[2] - 7, y), fill=(235, 150, 135), width=1)

    for idx, product in enumerate(products[:4]):
        px = basket[0] + int(width * 0.025) + idx * int(width * 0.035)
        py = basket[1] - int(height * 0.055) + (idx % 2) * int(height * 0.032)
        _draw_product(draw, px, py, int(width * 0.052), int(height * 0.105), product, small_font, small_font)

    receipt = (int(width * 0.7), int(height * 0.6), int(width * 0.91), int(height * 0.84))
    draw.rectangle(receipt, fill=(240, 237, 222), outline=(155, 148, 126), width=2)
    draw.text((receipt[0] + 8, receipt[1] + 8), "SCANNED", fill=(44, 48, 50), font=font)
    total = 0
    for idx, product in enumerate(products[:4]):
        price = _price(product)
        total += price
        y = receipt[1] + 34 + idx * 20
        draw.text((receipt[0] + 8, y), product["name"][:8], fill=(50, 54, 55), font=small_font)
        draw.text((receipt[2] - 42, y), f"${price}", fill=(50, 54, 55), font=small_font)
    draw.text((receipt[0] + 8, receipt[3] - 26), f"TOTAL ${total}", fill=(44, 48, 50), font=font)
    _draw_report_badge(draw, width, height, small_font)


def _draw_low_poly_store_room(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    back_wall: tuple[int, int, int] = (142, 163, 147),
    right_wall: tuple[int, int, int] = (181, 195, 149),
) -> dict[str, int]:
    ceiling_y = int(height * 0.28)
    floor_y = int(height * 0.58)
    vp_x = width // 2

    draw.rectangle((0, 0, width, ceiling_y), fill=(94, 94, 96))
    draw.polygon([(0, ceiling_y), (int(width * 0.18), floor_y), (0, height)], fill=(122, 144, 132))
    draw.polygon([(width, ceiling_y), (int(width * 0.82), floor_y), (width, height)], fill=right_wall)
    draw.polygon([(0, ceiling_y), (width, ceiling_y), (int(width * 0.82), floor_y), (int(width * 0.18), floor_y)], fill=back_wall)
    draw.polygon([(int(width * 0.18), floor_y), (int(width * 0.82), floor_y), (width, height), (0, height)], fill=(33, 30, 34))

    for x in (int(width * 0.22), int(width * 0.38), int(width * 0.62), int(width * 0.78)):
        draw.line((vp_x, ceiling_y, x, 0), fill=(72, 72, 74), width=1)
    for row in range(4):
        y = int(ceiling_y * (0.18 + row * 0.2))
        draw.line((0, y, width, y), fill=(76, 76, 78), width=1)

    _draw_ceiling_lights(draw, width, height, ceiling_y)
    return {"ceiling_y": ceiling_y, "floor_y": floor_y}


def _draw_ceiling_lights(draw: ImageDraw.ImageDraw, width: int, height: int, ceiling_y: int) -> None:
    for row in range(5):
        y = int(ceiling_y * (0.12 + row * 0.17))
        scale = 0.6 + row * 0.16
        light_w = int(width * 0.075 * scale)
        light_h = max(7, int(height * 0.025 * scale))
        spacing = int(width * 0.13 * scale)
        for col in range(-3, 4):
            cx = width // 2 + col * spacing
            if cx < -light_w or cx > width + light_w:
                continue
            skew = int(light_w * 0.16)
            draw.polygon(
                [
                    (cx - light_w // 2 + skew, y),
                    (cx + light_w // 2 + skew, y),
                    (cx + light_w // 2 - skew, y + light_h),
                    (cx - light_w // 2 - skew, y + light_h),
                ],
                fill=(242, 243, 238),
                outline=(218, 218, 214),
            )


def _draw_sari_shelf(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    base_y: int,
    width: int,
    height: int,
    label: str,
    products: list[dict],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x0 = center_x - width // 2
    x1 = center_x + width // 2
    y0 = base_y - height
    y1 = base_y
    side = max(8, width // 5)

    draw.rectangle((x0, y0, x1, y1), fill=(39, 36, 42), outline=(20, 18, 22), width=2)
    draw.polygon([(x1, y0), (x1 + side, y0 - side // 2), (x1 + side, y1 - side // 3), (x1, y1)], fill=(75, 83, 88), outline=(42, 48, 52))
    draw.polygon([(x0, y0), (x1, y0), (x1 + side, y0 - side // 2), (x0 + side, y0 - side // 2)], fill=(92, 97, 100), outline=(50, 55, 58))

    label_w = int(width * 0.62)
    draw.rectangle((center_x - label_w // 2, y0 + 8, center_x + label_w // 2, y0 + 26), fill=(31, 29, 35))
    draw.text((center_x - label_w // 2 + 6, y0 + 10), label, fill=(232, 232, 226), font=small_font)

    for row in range(2):
        shelf_y = y0 + int(height * (0.42 + row * 0.28))
        draw.line((x0 + 8, shelf_y, x1 - 8, shelf_y), fill=(200, 204, 198), width=max(2, width // 45))
        for col in range(4):
            product = products[(row * 4 + col) % len(products)]
            px = x0 + 12 + col * max(8, (width - 24) // 4)
            py = shelf_y - max(9, height // 7)
            pw = max(7, width // 8)
            ph = max(13, height // 5)
            draw.rectangle((px, py, px + pw, py + ph), fill=_category_color(product["category"]), outline=(12, 12, 14))
            draw.rectangle((px + 2, py + 3, px + pw - 2, py + max(5, ph // 3)), fill=(238, 234, 210))

    shadow = [(x0 - 5, y1), (x1 + side, y1 - side // 3), (x1 + side + 16, y1 + 12), (x0 + 8, y1 + 18)]
    draw.polygon(shadow, fill=(23, 20, 24))


def _draw_front_shelf(draw: ImageDraw.ImageDraw, products: list[dict], width: int, height: int) -> None:
    font = _font(max(11, width // 62))
    title_font = _font(max(18, width // 34), bold=True)
    small_font = _font(max(9, width // 80))

    _draw_header(draw, width, "RETAIL SHELF DEMO", "Aisle 03 - grocery products", title_font, font)
    shelf_x0, shelf_y0 = int(width * 0.1), int(height * 0.2)
    shelf_x1, shelf_y1 = int(width * 0.9), int(height * 0.83)
    draw.rectangle((shelf_x0, shelf_y0, shelf_x1, shelf_y1), fill=(55, 65, 70), outline=(30, 36, 39), width=3)

    levels = 4
    level_gap = (shelf_y1 - shelf_y0) // levels
    product_idx = 0
    for level in range(levels):
        y_base = shelf_y0 + (level + 1) * level_gap - 10
        draw.rectangle((shelf_x0 + 12, y_base, shelf_x1 - 12, y_base + 12), fill=(217, 222, 220))
        draw.rectangle((shelf_x0 + 12, y_base + 12, shelf_x1 - 12, y_base + 26), fill=(236, 227, 184))
        draw.text((shelf_x0 + 20, y_base + 4), f"LEVEL {level + 1}", fill=(62, 56, 43), font=small_font)
        for col in range(6):
            product = products[product_idx % len(products)]
            product_idx += 1
            px = shelf_x0 + 34 + col * ((shelf_x1 - shelf_x0 - 80) // 6)
            py = y_base - int(level_gap * 0.62)
            _draw_product(draw, px, py, int(width * 0.075), int(height * 0.12), product, font, small_font)

    draw.rectangle((shelf_x0 - 10, shelf_y1 + 8, shelf_x1 + 10, shelf_y1 + 28), fill=(42, 48, 52))
    _draw_report_badge(draw, width, height, font)


def _draw_retail_aisle(draw: ImageDraw.ImageDraw, products: list[dict], width: int, height: int) -> None:
    font = _font(max(11, width // 62))
    title_font = _font(max(18, width // 34), bold=True)
    small_font = _font(max(9, width // 80))

    _draw_header(draw, width, "SIMPLE RETAIL AISLE", "Stocked shelves and clear navigation lane", title_font, font)
    floor_y = int(height * 0.82)
    draw.polygon([(0, floor_y), (width, floor_y), (width, height), (0, height)], fill=(196, 184, 158))
    draw.polygon([(int(width * 0.44), int(height * 0.22)), (int(width * 0.56), int(height * 0.22)), (int(width * 0.72), floor_y), (int(width * 0.28), floor_y)], fill=(224, 221, 211))
    draw.line((int(width * 0.5), int(height * 0.22), int(width * 0.5), floor_y), fill=(186, 180, 166), width=2)

    left_shelf = [(0, int(height * 0.28)), (int(width * 0.36), int(height * 0.18)), (int(width * 0.28), floor_y), (0, height)]
    right_shelf = [(width, int(height * 0.28)), (int(width * 0.64), int(height * 0.18)), (int(width * 0.72), floor_y), (width, height)]
    draw.polygon(left_shelf, fill=(67, 77, 79), outline=(34, 42, 44))
    draw.polygon(right_shelf, fill=(67, 77, 79), outline=(34, 42, 44))

    for side in ("left", "right"):
        for row in range(4):
            y = int(height * (0.34 + row * 0.11))
            if side == "left":
                draw.line((12, y + row * 10, int(width * 0.31), y - 16), fill=(225, 230, 227), width=4)
                x_start = 20
            else:
                draw.line((width - 12, y + row * 10, int(width * 0.69), y - 16), fill=(225, 230, 227), width=4)
                x_start = int(width * 0.71)
            for col in range(4):
                product = products[(row * 4 + col) % len(products)]
                color = _category_color(product["category"])
                x = x_start + col * int(width * 0.05)
                box = (x, y - 30, x + int(width * 0.035), y + 8)
                draw.rectangle(box, fill=color, outline=(30, 30, 30))
                draw.rectangle((box[0] + 3, box[1] + 5, box[2] - 3, box[1] + 14), fill=(248, 245, 230))

    sign_x0, sign_y0 = int(width * 0.36), int(height * 0.12)
    draw.rectangle((sign_x0, sign_y0, int(width * 0.64), sign_y0 + 42), fill=(28, 40, 45))
    draw.text((sign_x0 + 18, sign_y0 + 10), "AISLE 03", fill=(250, 248, 235), font=title_font)
    draw.text((int(width * 0.39), floor_y + 20), "navigation lane", fill=(73, 74, 68), font=font)
    _draw_report_badge(draw, width, height, small_font)


def _draw_checkout_counter(draw: ImageDraw.ImageDraw, products: list[dict], width: int, height: int) -> None:
    font = _font(max(11, width // 62))
    title_font = _font(max(18, width // 34), bold=True)
    small_font = _font(max(9, width // 80))

    _draw_header(draw, width, "CHECKOUT COUNTER", "Scanner, basket, and itemized product trace", title_font, font)
    counter = (int(width * 0.08), int(height * 0.42), int(width * 0.92), int(height * 0.84))
    draw.rectangle(counter, fill=(88, 96, 98), outline=(42, 48, 50), width=3)
    draw.rectangle((counter[0], counter[1], counter[2], counter[1] + int(height * 0.09)), fill=(214, 219, 214), outline=(154, 160, 155))

    scanner = (int(width * 0.46), int(height * 0.46), int(width * 0.58), int(height * 0.56))
    draw.rounded_rectangle(scanner, radius=8, fill=(36, 48, 55), outline=(13, 22, 26), width=2)
    draw.line((scanner[0] + 10, scanner[1] + 18, scanner[2] - 10, scanner[3] - 18), fill=(75, 202, 140), width=3)
    draw.text((scanner[0] + 8, scanner[3] + 6), "scanner", fill=(24, 32, 36), font=small_font)

    basket = (int(width * 0.12), int(height * 0.54), int(width * 0.34), int(height * 0.77))
    draw.rectangle(basket, fill=(209, 79, 66), outline=(112, 41, 34), width=3)
    for x in range(basket[0] + 14, basket[2], 18):
        draw.line((x, basket[1] + 8, x, basket[3] - 8), fill=(245, 171, 158), width=2)
    for y in range(basket[1] + 18, basket[3], 18):
        draw.line((basket[0] + 8, y, basket[2] - 8, y), fill=(245, 171, 158), width=2)

    for idx, product in enumerate(products[:4]):
        px = basket[0] + 18 + idx * 28
        py = basket[1] - 24 + (idx % 2) * 20
        _draw_product(draw, px, py, int(width * 0.055), int(height * 0.105), product, small_font, small_font)

    receipt = (int(width * 0.66), int(height * 0.43), int(width * 0.88), int(height * 0.78))
    draw.rectangle(receipt, fill=(250, 248, 238), outline=(166, 160, 140), width=2)
    draw.text((receipt[0] + 12, receipt[1] + 10), "SCANNED ITEMS", fill=(36, 44, 47), font=font)
    total = 0
    for idx, product in enumerate(products[:5]):
        y = receipt[1] + 40 + idx * 24
        price = _price(product)
        total += price
        draw.text((receipt[0] + 12, y), product["name"][:9], fill=(52, 58, 60), font=small_font)
        draw.text((receipt[2] - 54, y), f"${price}", fill=(52, 58, 60), font=small_font)
    draw.line((receipt[0] + 10, receipt[3] - 38, receipt[2] - 10, receipt[3] - 38), fill=(160, 153, 130), width=2)
    draw.text((receipt[0] + 12, receipt[3] - 28), "TOTAL", fill=(36, 44, 47), font=font)
    draw.text((receipt[2] - 64, receipt[3] - 28), f"${total}", fill=(36, 44, 47), font=font)
    _draw_report_badge(draw, width, height, small_font)


def _draw_header(draw: ImageDraw.ImageDraw, width: int, title: str, subtitle: str, title_font: ImageFont.ImageFont, font: ImageFont.ImageFont) -> None:
    draw.rectangle((0, 0, width, 70), fill=(30, 43, 49))
    draw.text((28, 14), title, fill=(251, 248, 232), font=title_font)
    draw.text((30, 44), subtitle, fill=(197, 215, 210), font=font)


def _draw_product(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    product: dict,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    color = _category_color(product["category"])
    draw.rectangle((x, y, x + width, y + height), fill=color, outline=(34, 38, 40), width=2)
    draw.rectangle((x + 5, y + 8, x + width - 5, y + int(height * 0.48)), fill=(249, 246, 228), outline=(220, 214, 190))
    draw.text((x + 8, y + 12), product["name"][:10], fill=(38, 44, 46), font=font)
    draw.text((x + 8, y + 30), product["brand"][:11], fill=(86, 82, 72), font=small_font)
    draw.rectangle((x + 6, y + height - 28, x + width - 6, y + height - 8), fill=(255, 245, 184))
    draw.text((x + 10, y + height - 26), f"${_price(product)}", fill=(52, 44, 22), font=small_font)
    _draw_barcode(draw, x + width - 28, y + height - 28, 18, 18, product["barcode"])


def _draw_barcode(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, code: str) -> None:
    draw.rectangle((x, y, x + width, y + height), fill=(250, 250, 247))
    digits = [int(ch) for ch in code if ch.isdigit()] or [1, 0, 1, 0, 1]
    cursor = x + 2
    for idx, digit in enumerate(digits[:8]):
        bar_width = 1 + digit % 3
        if idx % 2 == 0 or digit % 2 == 1:
            draw.rectangle((cursor, y + 2, cursor + bar_width, y + height - 3), fill=(30, 30, 30))
        cursor += bar_width + 1
        if cursor >= x + width - 1:
            break


def _draw_report_badge(draw: ImageDraw.ImageDraw, width: int, height: int, font: ImageFont.ImageFont) -> None:
    text = "synthetic report scene"
    x0, y0 = int(width * 0.68), height - 36
    draw.rounded_rectangle((x0, y0, width - 24, height - 12), radius=8, fill=(30, 43, 49))
    draw.text((x0 + 12, y0 + 6), text, fill=(235, 238, 231), font=font)


def _category_color(category: str) -> tuple[int, int, int]:
    return CATEGORY_COLORS.get(category, (120, 132, 150))


def _price(product: dict) -> int:
    try:
        return int(product.get("price", 5))
    except (TypeError, ValueError):
        return 5


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
