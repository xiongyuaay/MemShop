from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


SingleShelfConfig = Dict[str, Any]


def load_single_shelf_config(path: str | Path) -> SingleShelfConfig:
    config_path = Path(path)
    if not config_path.exists():
        return {"enabled": False}
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return normalize_single_shelf_config(raw_config)


def normalize_single_shelf_config(config: SingleShelfConfig | None) -> SingleShelfConfig:
    if not config:
        return {"enabled": False}

    normalized = dict(config)
    normalized["enabled"] = bool(normalized.get("enabled", False))
    if not normalized["enabled"]:
        return normalized

    missing = [field for field in ("scene", "shelf_id") if not normalized.get(field)]
    if missing:
        raise ValueError(f"single shelf config missing required fields: {', '.join(missing)}")

    allowed_types = normalized.get("allowed_object_types") or []
    normalized["allowed_object_types"] = list(allowed_types)
    return normalized


def filter_catalog_to_single_shelf(
    catalog: List[Dict[str, Any]],
    config: SingleShelfConfig | None,
) -> List[Dict[str, Any]]:
    normalized = normalize_single_shelf_config(config)
    if not normalized.get("enabled", False):
        return list(catalog)

    scene = normalized["scene"]
    shelf_id = normalized["shelf_id"]
    source_object_id = normalized.get("source_object_id")
    allowed_types = set(normalized.get("allowed_object_types") or [])
    filtered = [
        product
        for product in catalog
        if product.get("scene") == scene
        and _matches_single_shelf(product, shelf_id, source_object_id)
        and (not allowed_types or product.get("object_type") in allowed_types)
    ]
    if not filtered:
        raise ValueError(
            "single shelf config matched no products: "
            f"scene={scene}, shelf_id={shelf_id}, source_object_id={source_object_id or ''}"
        )
    return filtered


def _matches_single_shelf(product: Dict[str, Any], shelf_id: str, source_object_id: str | None) -> bool:
    if not source_object_id:
        return product.get("shelf_id") == shelf_id

    product_source = product.get("source_receptacle_id")
    if product_source:
        return product_source == source_object_id
    return product.get("shelf_id") == shelf_id
