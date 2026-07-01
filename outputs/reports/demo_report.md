# Retail THOR Demo Report

This report is generated from local episode JSON files.

## Summary

- Episodes: 20
- Success: 17

## find_product

- Episode: `episode_000000`
- Instruction: 帮我找一个健康食品
- Target: `{"constraint": {"attribute": "healthy"}, "candidate_product_ids": ["apple"], "target_product_id": "apple", "target_object_id": "Apple|demo"}`
- Success: `True`
- Actions: `navigate_to_shelf, search_object, look_at, answer_user, finish`
- Screenshot: `images/episode_000000_step_000_rgb.png`

![episode_000000](images/episode_000000_step_000_rgb.png)

- Metrics: `{"target_found": true, "dialogue_resolved": false, "substitute_valid": false, "pickup_success": false, "placement_success": false, "num_high_level_steps": 5, "num_sim_steps": 5, "num_dialogue_turns": 0, "used_force_action": false, "final_distance_to_target": null}`

## dialogue_find_or_substitute

- Episode: `episode_000001`
- Instruction: 如果没有酸奶，帮我找一个健康食品
- Target: `{"constraint": {}, "candidate_product_ids": ["apple"], "requested_product": "酸奶", "missing_reason": "not_in_scene", "substitute_product_id": "apple", "substitute_reason": "shared attributes healthy", "target_product_id": "apple", "target_object_id": "Apple|demo", "substitute_object_id": "Apple|demo"}`
- Success: `True`
- Actions: `ask_npc, recommend_substitute, navigate_to_shelf, search_object, look_at, answer_user, finish`
- NPC dialogue: 当前没有酸奶，我会找一个健康替代品。
- Screenshot: `images/episode_000001_step_000_rgb.png`

![episode_000001](images/episode_000001_step_000_rgb.png)

- Metrics: `{"target_found": true, "dialogue_resolved": true, "substitute_valid": true, "pickup_success": false, "placement_success": false, "num_high_level_steps": 7, "num_sim_steps": 7, "num_dialogue_turns": 1, "used_force_action": false, "final_distance_to_target": null}`

## pick_and_place

- Episode: `episode_000002`
- Instruction: 帮我把苹果放到购物栏
- Target: `{"constraint": {"category": "fruit"}, "candidate_product_ids": ["apple"], "placement": "cart", "target_product_id": "apple", "target_object_id": "Apple|demo"}`
- Success: `True`
- Actions: `navigate_to_shelf, search_object, look_at, answer_user, pick_object, place_object, finish`
- Screenshot: `images/episode_000002_step_000_rgb.png`

![episode_000002](images/episode_000002_step_000_rgb.png)

- Metrics: `{"target_found": true, "dialogue_resolved": false, "substitute_valid": false, "pickup_success": true, "placement_success": true, "num_high_level_steps": 7, "num_sim_steps": 7, "num_dialogue_turns": 0, "used_force_action": false, "final_distance_to_target": null}`

