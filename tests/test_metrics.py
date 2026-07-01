from retail_thor.metrics import check_acceptance_criteria, summarize_episode_metrics


def test_summarize_episode_metrics_reports_success_flags_and_counts():
    episode = {
        "task_type": "dialogue_find_or_substitute",
        "high_level_plan": [
            {"action": "ask_npc", "sim_actions": [{"action": "Done"}]},
            {"action": "recommend_substitute", "sim_actions": []},
            {"action": "pick_object", "success": True, "sim_actions": [{"action": "PickupObject"}]},
            {"action": "place_object", "success": False, "sim_actions": [{"action": "PutObject"}]},
        ],
        "npc_dialogue": [{"speaker": "robot"}, {"speaker": "npc"}],
        "success": False,
    }

    metrics = summarize_episode_metrics(episode)

    assert metrics["dialogue_resolved"] is True
    assert metrics["substitute_valid"] is True
    assert metrics["pickup_success"] is True
    assert metrics["placement_success"] is False
    assert metrics["num_high_level_steps"] == 4
    assert metrics["num_sim_steps"] == 3
    assert metrics["num_dialogue_turns"] == 2


def test_check_acceptance_criteria_reports_dataset_gaps():
    episodes = [
        {"task_type": "find_product", "success": True, "high_level_plan": []},
        {
            "task_type": "dialogue_find_or_substitute",
            "success": True,
            "high_level_plan": [{"action": "recommend_substitute"}],
        },
        {
            "task_type": "pick_and_place",
            "success": True,
            "high_level_plan": [{"action": "pick_object"}, {"action": "place_object"}],
            "metrics": {"placement_success": True},
        },
    ]
    criteria = {
        "min_episodes": 4,
        "required_task_types": ["find_product", "dialogue_find_or_substitute", "pick_and_place"],
        "min_success": 3,
        "min_substitute_samples": 2,
        "min_pick_and_place_success": 1,
    }

    report = check_acceptance_criteria(episodes, criteria)

    assert report["passed"] is False
    assert report["counts"]["episodes"] == 3
    assert "min_episodes" in report["failed_checks"]
    assert "min_substitute_samples" in report["failed_checks"]
