import pytest

from retail_thor.episode_schema import (
    ALLOWED_ACTIONS,
    ALLOWED_FAILURE_REASONS,
    validate_success_conditions,
    validate_episode,
)


def minimal_episode():
    return {
        "episode_id": "episode_000001",
        "schema_version": "0.1.0",
        "backend": "ai2thor",
        "scene": "FloorPlan1",
        "random_seed": 0,
        "capability_family": ["object_loco_navigation"],
        "task_type": "find_product",
        "prompt": "帮我找一个水果",
        "target": {"category": "fruit", "target_object_id": "Apple|1"},
        "episode_idx": 1,
        "max_steps": 20,
        "instruction": "帮我找一个水果",
        "initial_state": {"agent_pose": {}, "held_object": None},
        "product_catalog": [],
        "shelf_graph": [],
        "npc_dialogue": [],
        "high_level_plan": [
            {
                "action": "look_at",
                "args": {"object_id": "Apple|1"},
                "preconditions": [],
                "expected_effects": [],
                "sim_actions": [{"action": "Teleport"}],
                "success": True,
                "failure_reason": None,
            }
        ],
        "sim_action_trace": [{"action": "Done", "success": True}],
        "observations": [
            {
                "step_idx": 0,
                "rgb_path": "outputs/images/e.png",
                "depth_path": None,
                "segmentation_path": None,
                "metadata_path": "outputs/episodes/e_meta.json",
                "agent_pose": {},
                "held_object": None,
            }
        ],
        "metrics": {"target_found": True},
        "success": True,
        "failure_reason": None,
        "provenance": {"generator": "test"},
    }


def test_action_and_failure_reason_enums_cover_required_behavior():
    assert {
        "navigate_to_shelf",
        "open_receptacle",
        "search_object",
        "look_at",
        "ask_npc",
        "answer_user",
        "confirm_product",
        "recommend_substitute",
        "pick_object",
        "place_object",
        "compare_objects",
        "finish",
    }.issubset(ALLOWED_ACTIONS)
    assert "pickup_failed" in ALLOWED_FAILURE_REASONS
    assert "placement_failed" in ALLOWED_FAILURE_REASONS


def test_validate_episode_accepts_minimal_valid_episode():
    validate_episode(minimal_episode())


def test_validate_episode_rejects_missing_required_field():
    episode = minimal_episode()
    episode.pop("instruction")

    with pytest.raises(ValueError, match="instruction"):
        validate_episode(episode)


def test_validate_episode_rejects_unknown_high_level_action():
    episode = minimal_episode()
    episode["high_level_plan"][0]["action"] = "dance"

    with pytest.raises(ValueError, match="Unknown high-level action"):
        validate_episode(episode)


def test_success_conditions_require_task_specific_evidence():
    find_episode = minimal_episode()
    find_episode["task_type"] = "find_product"
    find_episode["target"] = {"target_product_id": "apple"}
    find_episode["high_level_plan"] = [
        {
            "action": "look_at",
            "args": {"object_id": "apple"},
            "preconditions": [],
            "expected_effects": [],
            "sim_actions": [{"action": "Teleport"}],
            "success": True,
            "failure_reason": None,
        }
    ]

    validate_success_conditions(find_episode)

    find_episode["high_level_plan"][0]["args"]["object_id"] = "bread"
    with pytest.raises(ValueError, match="target_product_id"):
        validate_success_conditions(find_episode)


def test_pick_and_place_success_requires_pick_and_place_steps():
    episode = minimal_episode()
    episode["task_type"] = "pick_and_place"
    episode["success"] = True
    episode["high_level_plan"] = [
        {
            "action": "pick_object",
            "args": {"object_id": "apple"},
            "preconditions": [],
            "expected_effects": [],
            "sim_actions": [{"action": "PickupObject"}],
            "success": True,
            "failure_reason": None,
        }
    ]

    with pytest.raises(ValueError, match="place_object"):
        validate_success_conditions(episode)
