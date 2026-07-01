from __future__ import annotations

import json

from retail_thor.npc import CustomerNPC, GPT4oMiniNPCPlanner, NPCPlannerResponse
from retail_thor.world_knowledge_manager import WorldKnowledgeManager


CATALOG = [
    {
        "product_id": "chocolate_red_001",
        "display_name_en": "red chocolate bar",
        "display_name_zh": "红色巧克力",
        "category": "snack",
        "attributes": ["red", "packaged", "sweet", "chocolate"],
        "brand": "Sweet Lab",
        "price": 6,
        "shelf_id": "shelf_snack_01",
        "object_id": "Chocolate|1",
    },
    {
        "product_id": "chips_red_001",
        "display_name_en": "red chips",
        "display_name_zh": "红色薯片",
        "category": "snack",
        "attributes": ["red", "packaged", "salty", "chips"],
        "brand": "Crunch Lab",
        "price": 5,
        "shelf_id": "shelf_snack_02",
        "object_id": "Chips|1",
    },
]


class StaticNPCPlanner:
    model = "gpt-4o-mini"

    def __init__(self):
        self.calls = []

    def respond(self, *, wkm, target_product_id, initial_request, agent_utterance, dialogue_history):
        self.calls.append(
            {
                "target_product_id": target_product_id,
                "initial_request": initial_request,
                "agent_utterance": agent_utterance,
                "dialogue_history": list(dialogue_history),
            }
        )
        product = wkm.get_product(target_product_id)
        return NPCPlannerResponse(
            utterance="Chocolate.",
            rationale=f"Target product is {product['display_name_en']}.",
            product_id=target_product_id,
            wkm_calls=wkm.consume_call_log(),
        )


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "FakeResponse",
            (),
            {
                "output_text": json.dumps(
                    {
                        "utterance": "Chocolate.",
                        "rationale": "The hidden target product has the chocolate attribute.",
                    }
                )
            },
        )()


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_world_knowledge_manager_exposes_product_lookup_and_search_api():
    wkm = WorldKnowledgeManager(CATALOG)

    product = wkm.get_product("chocolate_red_001")
    candidates = wkm.search_products({"category": "snack", "attributes": ["red", "packaged"]})
    chocolate_attrs = wkm.get_candidate_attributes(["chocolate_red_001"])

    assert product["display_name_en"] == "red chocolate bar"
    assert {item["product_id"] for item in candidates} == {"chocolate_red_001", "chips_red_001"}
    assert chocolate_attrs == [
        {
            "product_id": "chocolate_red_001",
            "display_name_en": "red chocolate bar",
            "display_name_zh": "红色巧克力",
            "category": "snack",
            "attributes": ["red", "packaged", "sweet", "chocolate"],
            "brand": "Sweet Lab",
            "shelf_id": "shelf_snack_01",
        }
    ]


def test_customer_npc_uses_wkm_and_planner_to_answer_agent_question_without_actions():
    wkm = WorldKnowledgeManager(CATALOG)
    planner = StaticNPCPlanner()
    npc = CustomerNPC(
        initial_request="I want a red packaged sweet snack.",
        target_product_id="chocolate_red_001",
        wkm=wkm,
        planner=planner,
    )

    first_turn = npc.initial_utterance()
    response = npc.respond("Is it chocolate or chips?")

    assert first_turn == "I want a red packaged sweet snack."
    assert response.utterance == "Chocolate."
    assert not hasattr(response, "action")
    assert response.wkm_calls[0]["api"] == "get_product"
    assert planner.calls[0]["agent_utterance"] == "Is it chocolate or chips?"
    assert npc.dialogue_history == [
        {"speaker": "npc", "utterance": "I want a red packaged sweet snack."},
        {"speaker": "agent", "utterance": "Is it chocolate or chips?"},
        {"speaker": "npc", "utterance": "Chocolate."},
    ]


def test_gpt4o_mini_npc_planner_uses_target_product_info_from_wkm():
    wkm = WorldKnowledgeManager(CATALOG)
    client = FakeOpenAIClient()
    planner = GPT4oMiniNPCPlanner(client=client)

    response = planner.respond(
        wkm=wkm,
        target_product_id="chocolate_red_001",
        initial_request="I want a red packaged sweet snack.",
        agent_utterance="Is it chocolate or chips?",
        dialogue_history=[],
    )

    assert planner.model == "gpt-4o-mini"
    assert response.utterance == "Chocolate."
    assert response.product_id == "chocolate_red_001"
    assert response.wkm_calls[0]["api"] == "get_product"

    call = client.responses.calls[0]
    assert call["model"] == "gpt-4o-mini"
    prompt = call["input"][0]["content"][0]["text"]
    assert "red chocolate bar" in prompt
    assert "Is it chocolate or chips?" in prompt
    assert "Do not output embodied actions" in prompt
