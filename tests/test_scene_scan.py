from retail_thor.scene_scan import build_scene_inventory, build_shelf_regions, choose_cart_receptacle, normalize_object


def test_normalize_object_keeps_required_ai2thor_fields():
    normalized = normalize_object(
        {
            "objectId": "Apple|1",
            "objectType": "Apple",
            "name": "Apple_1",
            "pickupable": True,
            "parentReceptacles": ["CounterTop|1"],
            "axisAlignedBoundingBox": {"center": {"x": 1}},
        }
    )

    assert normalized["objectId"] == "Apple|1"
    assert normalized["pickupable"] is True
    assert normalized["parentReceptacles"] == ["CounterTop|1"]
    assert "objectOrientedBoundingBox" in normalized


def test_build_shelf_regions_maps_receptacles_to_semantic_shelves():
    shelves = build_shelf_regions(
        "FloorPlan1",
        [
            {
                "objectId": "Fridge|1",
                "objectType": "Fridge",
                "receptacle": True,
                "openable": True,
                "visible": False,
                "receptacleObjectIds": ["Tomato|1"],
                "position": {"x": 1, "y": 0, "z": 2},
            },
            {"objectId": "Apple|1", "objectType": "Apple", "receptacle": False},
        ],
    )

    assert len(shelves) == 1
    assert shelves[0]["shelf_id"] == "shelf_FloorPlan1_000"
    assert shelves[0]["source_object_id"] == "Fridge|1"
    assert shelves[0]["contains_object_ids"] == ["Tomato|1"]
    assert shelves[0]["semantic_zone"] == "cold_shelf"


def test_choose_cart_receptacle_prefers_real_receptacle_and_falls_back_to_abstract_cart():
    real = choose_cart_receptacle(
        "FloorPlan1",
        [{"objectId": "Bowl|1", "objectType": "Bowl", "receptacle": True, "position": {"x": 0}}],
    )
    fallback = choose_cart_receptacle("FloorPlan1", [{"objectId": "Apple|1", "objectType": "Apple"}])

    assert real["cart_receptacle_id"] == "Bowl|1"
    assert real["is_abstract"] is False
    assert fallback["cart_receptacle_id"] is None
    assert fallback["is_abstract"] is True


def test_build_scene_inventory_records_randomization_metadata_and_stats():
    inventory = build_scene_inventory(
        "FloorPlan1",
        [
            {
                "objectId": "CounterTop|1",
                "objectType": "CounterTop",
                "receptacle": True,
                "pickupable": False,
                "receptacleObjectIds": ["Apple|1"],
            },
            {
                "objectId": "Apple|1",
                "objectType": "Apple",
                "pickupable": True,
                "receptacle": False,
                "parentReceptacles": ["CounterTop|1"],
            },
        ],
        randomization={
            "randomized": True,
            "random_seed": 7,
            "random_spawn_success": True,
            "error_message": "",
        },
    )

    assert inventory["scene"] == "FloorPlan1"
    assert inventory["stats"]["num_objects"] == 2
    assert inventory["stats"]["num_pickupable"] == 1
    assert inventory["stats"]["num_receptacles"] == 1
    assert inventory["stats"]["num_shelves"] == 1
    assert inventory["randomization"]["randomized"] is True
    assert inventory["randomization"]["random_seed"] == 7
    assert inventory["randomization"]["random_spawn_success"] is True
