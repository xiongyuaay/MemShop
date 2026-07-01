from retail_thor.single_shelf import filter_catalog_to_single_shelf


def test_filter_catalog_to_single_shelf_keeps_only_configured_scene_and_shelf():
    catalog = [
        {
            "product_id": "apple",
            "scene": "FloorPlan1",
            "shelf_id": "shelf_FloorPlan1_010",
            "object_type": "Apple",
        },
        {
            "product_id": "bread",
            "scene": "FloorPlan1",
            "shelf_id": "shelf_FloorPlan1_010",
            "object_type": "Bread",
        },
        {
            "product_id": "bottle",
            "scene": "FloorPlan1",
            "shelf_id": "shelf_FloorPlan1_023",
            "object_type": "Bottle",
        },
        {
            "product_id": "tomato",
            "scene": "FloorPlan2",
            "shelf_id": "shelf_FloorPlan1_010",
            "object_type": "Tomato",
        },
    ]
    config = {
        "enabled": True,
        "scene": "FloorPlan1",
        "shelf_id": "shelf_FloorPlan1_010",
        "source_object_id": "CounterTop|-00.08|+01.15|00.00",
    }

    filtered = filter_catalog_to_single_shelf(catalog, config)

    assert [product["product_id"] for product in filtered] == ["apple", "bread"]
    assert {product["scene"] for product in filtered} == {"FloorPlan1"}
    assert {product["shelf_id"] for product in filtered} == {"shelf_FloorPlan1_010"}


def test_filter_catalog_to_single_shelf_can_be_disabled():
    catalog = [
        {"product_id": "apple", "scene": "FloorPlan1", "shelf_id": "shelf_FloorPlan1_010"},
        {"product_id": "bottle", "scene": "FloorPlan1", "shelf_id": "shelf_FloorPlan1_023"},
    ]

    assert filter_catalog_to_single_shelf(catalog, {"enabled": False}) == catalog


def test_filter_catalog_to_single_shelf_prefers_stable_source_receptacle_id():
    catalog = [
        {
            "product_id": "bread",
            "scene": "FloorPlan1",
            "shelf_id": "shelf_FloorPlan1_999",
            "source_receptacle_id": "CounterTop|-00.08|+01.15|00.00",
        },
        {
            "product_id": "bottle",
            "scene": "FloorPlan1",
            "shelf_id": "shelf_FloorPlan1_010",
            "source_receptacle_id": "CounterTop|other",
        },
    ]
    config = {
        "enabled": True,
        "scene": "FloorPlan1",
        "shelf_id": "shelf_FloorPlan1_010",
        "source_object_id": "CounterTop|-00.08|+01.15|00.00",
    }

    filtered = filter_catalog_to_single_shelf(catalog, config)

    assert [product["product_id"] for product in filtered] == ["bread"]
