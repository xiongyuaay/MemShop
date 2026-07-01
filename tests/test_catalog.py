from retail_thor.catalog import (
    build_product_catalog,
    assert_catalog_complete,
    choose_substitute,
    match_product,
)


PRODUCT_RULES = {
    "Apple": {
        "display_name_zh": "苹果",
        "display_name_en": "apple",
        "category": "fruit",
        "attributes": ["fresh", "healthy", "low_sugar"],
        "synonyms_zh": ["水果"],
        "price_range": [3, 8],
        "brand_pool": ["本地果园"],
        "substitute_policy": ["same_category", "shared_attributes"],
        "is_food": True,
        "is_pickup_required": True,
    },
    "Tomato": {
        "display_name_zh": "番茄",
        "display_name_en": "tomato",
        "category": "vegetable",
        "attributes": ["fresh", "healthy"],
        "synonyms_zh": ["西红柿"],
        "price_range": [2, 6],
        "brand_pool": ["田园"],
        "substitute_policy": ["shared_attributes"],
        "is_food": True,
        "is_pickup_required": True,
    },
    "Knife": {
        "display_name_zh": "刀",
        "display_name_en": "knife",
        "category": "tool",
        "attributes": ["sharp"],
        "synonyms_zh": [],
        "price_range": [10, 30],
        "brand_pool": ["工具"],
        "substitute_policy": [],
        "is_food": False,
        "is_pickup_required": True,
        "exclude_from_catalog": True,
    },
}


OBJECTS = [
    {
        "objectId": "Apple|1|0|1",
        "objectType": "Apple",
        "name": "Apple_1",
        "position": {"x": 1.0, "y": 0.8, "z": 1.0},
        "pickupable": True,
        "visible": True,
        "parentReceptacles": ["CounterTop|0|0|0"],
    },
    {
        "objectId": "Tomato|2|0|1",
        "objectType": "Tomato",
        "name": "Tomato_1",
        "position": {"x": 2.0, "y": 0.8, "z": 1.0},
        "pickupable": True,
        "visible": False,
        "parentReceptacles": ["Fridge|0|0|0"],
    },
    {
        "objectId": "Knife|3|0|1",
        "objectType": "Knife",
        "name": "Knife_1",
        "position": {"x": 3.0, "y": 0.8, "z": 1.0},
        "pickupable": True,
        "visible": True,
        "parentReceptacles": ["CounterTop|0|0|0"],
    },
]


SHELVES = {
    "CounterTop|0|0|0": "shelf_FloorPlan1_000",
    "Fridge|0|0|0": "shelf_FloorPlan1_001",
}


def test_build_product_catalog_filters_excluded_and_uses_deterministic_prices():
    first = build_product_catalog("FloorPlan1", OBJECTS, PRODUCT_RULES, SHELVES, seed=7)
    second = build_product_catalog("FloorPlan1", OBJECTS, PRODUCT_RULES, SHELVES, seed=7)

    assert [item["object_type"] for item in first] == ["Apple", "Tomato"]
    assert [item["price"] for item in first] == [item["price"] for item in second]
    assert first[0]["product_id"] == "FloorPlan1_Apple_000"
    assert first[0]["shelf_id"] == "shelf_FloorPlan1_000"
    assert first[0]["display_name_zh"] == "苹果"
    assert first[0]["stock_status"] == "in_stock"


def test_match_product_supports_category_attribute_and_max_price():
    product = build_product_catalog("FloorPlan1", OBJECTS, PRODUCT_RULES, SHELVES, seed=7)[0]

    assert match_product(product, {"category": "fruit", "attribute": "healthy", "max_price": 8})
    assert not match_product(product, {"category": "drink"})
    assert not match_product(product, {"attribute": "breakfast"})
    assert not match_product(product, {"max_price": 1})


def test_choose_substitute_prefers_shared_attributes_when_exact_item_missing():
    catalog = build_product_catalog("FloorPlan1", OBJECTS, PRODUCT_RULES, SHELVES, seed=7)

    substitute = choose_substitute(
        catalog,
        requested={"display_name_zh": "酸奶", "category": "dairy", "attributes": ["healthy"]},
    )

    assert substitute is not None
    assert substitute["substitute_product_id"] in {"FloorPlan1_Apple_000", "FloorPlan1_Tomato_001"}
    assert "healthy" in substitute["substitute_reason"]


def test_assert_catalog_complete_rejects_missing_required_fields():
    catalog = build_product_catalog("FloorPlan1", OBJECTS, PRODUCT_RULES, SHELVES, seed=7)
    broken = [dict(catalog[0], shelf_id="")]

    assert_catalog_complete(catalog)

    try:
        assert_catalog_complete(broken)
    except ValueError as exc:
        assert "shelf_id" in str(exc)
    else:
        raise AssertionError("expected missing shelf_id to be rejected")


def test_build_product_catalog_skips_products_without_traceable_shelf():
    objects = [
        {
            "objectId": "Apple|unknown",
            "objectType": "Apple",
            "name": "Apple_unknown",
            "position": {"x": 0, "y": 0, "z": 0},
            "pickupable": True,
            "visible": True,
            "parentReceptacles": ["Floor|1"],
        }
    ]

    assert build_product_catalog("FloorPlan1", objects, PRODUCT_RULES, SHELVES, seed=7) == []


def test_assert_catalog_complete_rejects_unknown_shelf():
    catalog = build_product_catalog("FloorPlan1", OBJECTS, PRODUCT_RULES, SHELVES, seed=7)
    broken = [dict(catalog[0], shelf_id="unknown_shelf")]

    try:
        assert_catalog_complete(broken)
    except ValueError as exc:
        assert "unknown_shelf" in str(exc)
    else:
        raise AssertionError("expected unknown_shelf to be rejected")
