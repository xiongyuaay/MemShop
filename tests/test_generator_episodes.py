from retail_thor.episode_schema import validate_episode
from retail_thor.generator import build_episode


CATALOG = [
    {
        "product_id": "apple",
        "object_id": "Apple|1",
        "object_type": "Apple",
        "scene": "FloorPlan1",
        "shelf_id": "shelf_001",
        "display_name_zh": "苹果",
        "category": "fruit",
        "attributes": ["fresh", "healthy"],
        "brand": "每日鲜果",
        "price": 6,
        "stock_status": "in_stock",
        "pickupable": True,
        "visible_at_start": True,
        "position": {"x": 1, "y": 1, "z": 1},
        "scene_randomization": {"randomized": True, "random_seed": 3, "random_spawn_success": True},
    },
    {
        "product_id": "bread",
        "object_id": "Bread|1",
        "object_type": "Bread",
        "scene": "FloorPlan1",
        "shelf_id": "shelf_002",
        "display_name_zh": "面包",
        "category": "bakery",
        "attributes": ["breakfast"],
        "brand": "麦香",
        "price": 8,
        "stock_status": "in_stock",
        "pickupable": True,
        "visible_at_start": True,
        "position": {"x": 2, "y": 1, "z": 1},
    },
]


def test_build_find_product_episode_records_constraints_candidates_and_target():
    episode = build_episode(
        idx=0,
        task_type="find_product",
        template={"instruction": "帮我找一个水果", "constraint": {"category": "fruit"}},
        catalog=CATALOG,
        seed=0,
    )

    validate_episode(episode)
    assert episode["target"]["constraint"] == {"category": "fruit"}
    assert episode["target"]["candidate_product_ids"] == ["apple"]
    assert episode["target"]["target_product_id"] == "apple"
    assert episode["target"]["target_object_id"] == "Apple|1"
    assert {"shelf_id": "shelf_001", "source_object_id": "shelf_001"} in episode["shelf_graph"]
    assert episode["provenance"]["randomization"]["random_seed"] == 3
    assert "look_at" in [step["action"] for step in episode["high_level_plan"]]


def test_build_dialogue_episode_records_substitute_recommendation():
    episode = build_episode(
        idx=0,
        task_type="dialogue_find_or_substitute",
        template={"instruction": "如果没有酸奶，帮我找一个健康食品", "expected_dialogue": "substitute"},
        catalog=CATALOG,
        seed=0,
    )

    validate_episode(episode)
    actions = [step["action"] for step in episode["high_level_plan"]]
    assert episode["capability_family"] == ["social_loco_navigation"]
    assert episode["target"]["missing_reason"] == "not_in_scene"
    assert "ask_npc" in actions
    assert "recommend_substitute" in actions
    assert any(turn["intent"] == "recommend_substitute" for turn in episode["npc_dialogue"])


def test_build_pick_and_place_episode_contains_pick_and_place_actions():
    episode = build_episode(
        idx=0,
        task_type="pick_and_place",
        template={"instruction": "帮我把苹果放到购物栏", "constraint": {"category": "fruit"}, "placement": "cart"},
        catalog=CATALOG,
        seed=0,
    )

    validate_episode(episode)
    actions = [step["action"] for step in episode["high_level_plan"]]
    assert episode["capability_family"] == ["loco_manipulation"]
    assert "pick_object" in actions
    assert "place_object" in actions
    assert episode["target"]["placement"] == "cart"


def test_build_episode_keeps_negative_sample_when_no_product_matches():
    episode = build_episode(
        idx=0,
        task_type="find_product",
        template={"instruction": "帮我找一个玩具", "constraint": {"category": "toy"}},
        catalog=CATALOG,
        seed=0,
    )

    validate_episode(episode)
    assert episode["success"] is False
    assert episode["failure_reason"] == "no_matching_product"
    assert episode["target"]["candidate_product_ids"] == []
