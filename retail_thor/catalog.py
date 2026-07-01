from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, Iterable, List, Optional


Product = Dict[str, Any]


REQUIRED_CATALOG_FIELDS = {
    "product_id",
    "object_id",
    "object_type",
    "scene",
    "shelf_id",
    "display_name_zh",
    "category",
    "attributes",
    "brand",
    "price",
    "stock_status",
    "pickupable",
    "visible_at_start",
    "position",
}


def stable_price(scene: str, object_id: str, price_range: Iterable[int], seed: int) -> int:
    low, high = list(price_range)
    digest = hashlib.sha256(f"{scene}:{object_id}:{seed}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return rng.randint(low, high)


def parent_shelf_id(obj: Dict[str, Any], shelf_lookup: Dict[str, str]) -> str:
    parents = obj.get("parentReceptacles") or []
    for parent in parents:
        if parent in shelf_lookup:
            return shelf_lookup[parent]
    return "unknown_shelf"


def parent_receptacle_id(obj: Dict[str, Any], shelf_lookup: Dict[str, str]) -> str:
    parents = obj.get("parentReceptacles") or []
    for parent in parents:
        if parent in shelf_lookup:
            return parent
    return ""


def build_product_catalog(
    scene: str,
    objects: List[Dict[str, Any]],
    product_rules: Dict[str, Dict[str, Any]],
    shelf_lookup: Dict[str, str],
    seed: int = 0,
) -> List[Product]:
    catalog: List[Product] = []
    counts: Dict[str, int] = {}

    for obj in objects:
        if not obj.get("pickupable", False):
            continue
        object_type = obj.get("objectType")
        rule = product_rules.get(object_type)
        if not rule or rule.get("exclude_from_catalog", False):
            continue
        shelf_id = parent_shelf_id(obj, shelf_lookup)
        if shelf_id == "unknown_shelf":
            continue
        source_receptacle_id = parent_receptacle_id(obj, shelf_lookup)

        idx = counts.get(object_type, 0)
        counts[object_type] = idx + 1
        product_id = f"{scene}_{object_type}_{idx:03d}"

        catalog.append(
            {
                "product_id": product_id,
                "object_id": obj.get("objectId"),
                "object_type": object_type,
                "scene": scene,
                "shelf_id": shelf_id,
                "source_receptacle_id": source_receptacle_id,
                "display_name_zh": rule["display_name_zh"],
                "display_name_en": rule["display_name_en"],
                "category": rule["category"],
                "attributes": list(rule.get("attributes", [])),
                "synonyms_zh": list(rule.get("synonyms_zh", [])),
                "brand": _choose_brand(scene, obj.get("objectId", ""), rule.get("brand_pool", []), seed),
                "price": stable_price(scene, obj.get("objectId", ""), rule["price_range"], seed),
                "stock_status": obj.get("stock_status", "in_stock"),
                "pickupable": obj.get("pickupable", False),
                "visible_at_start": obj.get("visible", False),
                "position": obj.get("position", {}),
                "parent_receptacles": obj.get("parentReceptacles") or [],
                "is_food": rule.get("is_food", False),
                "is_pickup_required": rule.get("is_pickup_required", True),
            }
        )

    return catalog


def match_product(product: Product, constraint: Dict[str, Any]) -> bool:
    if "category" in constraint and product.get("category") != constraint["category"]:
        return False
    if "attribute" in constraint and constraint["attribute"] not in product.get("attributes", []):
        return False
    if "attributes" in constraint:
        required = set(constraint["attributes"])
        if not required.issubset(set(product.get("attributes", []))):
            return False
    if "max_price" in constraint and product.get("price", float("inf")) > constraint["max_price"]:
        return False
    if "pickupable" in constraint and product.get("pickupable") != constraint["pickupable"]:
        return False
    return True


def choose_substitute(catalog: List[Product], requested: Dict[str, Any]) -> Optional[Dict[str, str]]:
    requested_category = requested.get("category")
    requested_attributes = set(requested.get("attributes", []))
    best_product: Optional[Product] = None
    best_score = -1

    for product in catalog:
        if product.get("stock_status", "in_stock") != "in_stock":
            continue
        score = 0
        if requested_category and product.get("category") == requested_category:
            score += 10
        shared = requested_attributes.intersection(product.get("attributes", []))
        score += len(shared) * 3
        if product.get("is_food"):
            score += 1
        if score > best_score:
            best_score = score
            best_product = product

    if best_product is None or best_score <= 0:
        return None

    shared_attrs = sorted(requested_attributes.intersection(best_product.get("attributes", [])))
    reason_parts = []
    if best_product.get("category") == requested_category:
        reason_parts.append(f"same category {requested_category}")
    if shared_attrs:
        reason_parts.append("shared attributes " + ",".join(shared_attrs))

    return {
        "requested_product": requested.get("display_name_zh", requested.get("display_name_en", "unknown")),
        "missing_reason": requested.get("missing_reason", "not_in_scene"),
        "substitute_product_id": best_product["product_id"],
        "substitute_reason": "; ".join(reason_parts) or "best available product",
    }


def assert_catalog_complete(catalog: List[Product]) -> None:
    for idx, product in enumerate(catalog):
        missing = sorted(REQUIRED_CATALOG_FIELDS.difference(product))
        if missing:
            raise ValueError(f"catalog[{idx}] missing fields: {', '.join(missing)}")

        for field in ("product_id", "object_id", "object_type", "scene", "shelf_id", "display_name_zh", "category"):
            if product.get(field) in (None, ""):
                raise ValueError(f"catalog[{idx}] has empty {field}")

        if product.get("shelf_id") == "unknown_shelf":
            raise ValueError(f"catalog[{idx}] has unknown_shelf")
        if product.get("price") is None:
            raise ValueError(f"catalog[{idx}] has empty price")


def _choose_brand(scene: str, object_id: str, brand_pool: List[str], seed: int) -> str:
    if not brand_pool:
        return "generic"
    digest = hashlib.sha256(f"brand:{scene}:{object_id}:{seed}".encode("utf-8")).hexdigest()
    return brand_pool[int(digest[:8], 16) % len(brand_pool)]
