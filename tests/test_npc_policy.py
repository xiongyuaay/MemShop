from retail_thor.npc import resolve_dialogue_task


def test_npc_policy_asks_for_clarification_when_instruction_is_ambiguous():
    result = resolve_dialogue_task(
        instruction="帮我找一个吃的",
        catalog=[
            {"product_id": "p1", "category": "fruit", "attributes": ["healthy"], "display_name_zh": "苹果"},
            {"product_id": "p2", "category": "bakery", "attributes": ["breakfast"], "display_name_zh": "面包"},
        ],
    )

    assert result["requires_clarification"] is True
    assert result["dialogue"][0]["speaker"] == "npc"
    assert result["dialogue"][0]["intent"] == "request_product"
    assert result["dialogue"][1]["speaker"] == "agent"
    assert result["dialogue"][1]["intent"] == "clarify_product_constraint"
    assert all("action" not in turn for turn in result["dialogue"])


def test_npc_policy_recommends_substitute_when_requested_item_is_missing():
    result = resolve_dialogue_task(
        instruction="如果没有酸奶，帮我找一个健康食品",
        catalog=[
            {"product_id": "p1", "category": "fruit", "attributes": ["healthy"], "display_name_zh": "苹果"},
        ],
    )

    assert result["requires_clarification"] is False
    assert result["selected_product_id"] == "p1"
    assert result["dialogue"][0]["speaker"] == "npc"
    assert result["dialogue"][0]["intent"] == "request_product"
    assert any(turn["speaker"] == "agent" and turn["intent"] == "recommend_substitute" for turn in result["dialogue"])
    assert all("action" not in turn for turn in result["dialogue"])
