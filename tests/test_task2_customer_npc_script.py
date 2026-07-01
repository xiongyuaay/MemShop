from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "11_run_gpt4o_mini_fbx_customer_npc_task.py"
FORBIDDEN_NAMED_PRODUCT_TERMS = ("apple", "tomato", "bread", "mug", "fruit", "vegetable", "food")


class QueueResponses:
    def __init__(self):
        self.calls = []
        self.outputs = [
            {
                "decision_type": "ask_npc",
                "utterance": "Do you mean the cup-shaped red item or one of the round red items?",
                "action": "",
                "rationale": "The request is ambiguous between several red products in the current image.",
                "confidence": 0.74,
            },
            {
                "utterance": "The cup-shaped cylindrical one with a handle.",
                "rationale": "The hidden target is the cup-shaped cylindrical form.",
            },
            {
                "decision_type": "navigation_action",
                "utterance": "",
                "action": "MoveAhead",
                "rationale": "The target has been clarified as the red cup-shaped cylinder; move closer to the shelf.",
                "confidence": 0.76,
            },
            {
                "decision_type": "navigation_action",
                "utterance": "",
                "action": "Done",
                "rationale": "The clarified red cup-shaped cylinder is visible and close enough.",
                "confidence": 0.9,
            },
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.outputs.pop(0)
        return type("FakeResponse", (), {"output_text": json.dumps(payload)})()


class QueueClient:
    def __init__(self):
        self.responses = QueueResponses()


class FailAfterClarificationResponses:
    def __init__(self):
        self.calls = []
        self.outputs = [
            {
                "decision_type": "ask_npc",
                "utterance": "Which red product do you mean?",
                "action": "",
                "rationale": "The visible shelf contains multiple red products.",
                "confidence": 0.73,
            },
            {
                "utterance": "The cup-shaped cylindrical one with a handle.",
                "rationale": "The target is the cup-shaped cylindrical form.",
            },
            RuntimeError("simulated API failure"),
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.outputs.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return type("FakeResponse", (), {"output_text": json.dumps(payload)})()


class FailAfterClarificationClient:
    def __init__(self):
        self.responses = FailAfterClarificationResponses()


def _load_script_module():
    spec = importlib.util.spec_from_file_location("task2_customer_npc", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_no_named_product_terms(text: str) -> None:
    lowered = text.casefold()
    for term in FORBIDDEN_NAMED_PRODUCT_TERMS:
        assert term not in lowered


def test_task2_script_exposes_hardcoded_gpt4o_mini_and_clash_proxy_config():
    module = _load_script_module()
    source = SCRIPT.read_text(encoding="utf-8")

    assert module.OPENAI_MODEL == "gpt-4o-mini"
    assert module.CLASH_HTTP_PROXY == "http://127.0.0.1:10808"
    assert module.CLASH_HTTPS_PROXY == "http://127.0.0.1:10808"
    assert isinstance(module.OPENAI_API_KEY, str)
    assert module.OPENAI_API_KEY.strip()
    assert "os.environ.get(\"OPENAI_API_KEY\"" not in source


def test_task2_customer_npc_run_asks_when_ambiguous_then_navigates_to_done(tmp_path):
    module = _load_script_module()
    client = QueueClient()
    report = module.run_customer_npc_task(client=client, output_dir=tmp_path, width=480, height=320, max_steps=4)

    assert report["run_type"] == "real_gpt4o_mini_customer_npc_navigation_task"
    assert report["model"] == "gpt-4o-mini"
    assert report["success"] is True
    assert report["termination"] == {
        "status": "done",
        "reason": "embodied_agent_done_after_clarification",
        "done_action_required": True,
    }
    assert [turn["speaker"] for turn in report["dialogue"]] == ["npc", "agent", "npc"]
    assert report["dialogue"][0]["utterance"] == report["scenario"]["npc_initial_request"]
    assert report["dialogue"][1]["utterance"] == "Do you mean the cup-shaped red item or one of the round red items?"
    assert report["dialogue"][2]["utterance"] == "The cup-shaped cylindrical one with a handle."
    _assert_no_named_product_terms(json.dumps(report["scenario"], ensure_ascii=False))
    _assert_no_named_product_terms(json.dumps(report["dialogue"], ensure_ascii=False))
    assert all("action" not in turn for turn in report["dialogue"])
    assert report["dialogue"][2]["wkm_calls"][0]["api"] == "get_product"
    assert [step["decision_type"] for step in report["agent_decision_trace"]] == [
        "ask_npc",
        "navigation_action",
        "navigation_action",
    ]
    assert [step["action"] for step in report["navigation_trace"]] == ["MoveAhead", "Done"]
    assert (tmp_path / "images" / "step_000_input.png").exists()
    assert (tmp_path / "images" / "step_001_input.png").exists()
    assert (tmp_path / "images" / "step_002_input.png").exists()
    assert (tmp_path / "task2_customer_npc_run.json").exists()
    assert (tmp_path / "raw_responses" / "agent_step_000_response.json").exists()
    assert (tmp_path / "raw_responses" / "npc_step_000_response.json").exists()
    assert (tmp_path / "raw_responses" / "agent_step_001_response.json").exists()
    assert (tmp_path / "raw_responses" / "agent_step_002_response.json").exists()

    agent_ask_call, npc_call, agent_move_call, agent_done_call = client.responses.calls
    assert agent_ask_call["model"] == "gpt-4o-mini"
    assert npc_call["model"] == "gpt-4o-mini"
    assert agent_move_call["model"] == "gpt-4o-mini"
    assert agent_done_call["model"] == "gpt-4o-mini"
    assert any(item["type"] == "input_image" for item in agent_ask_call["input"][0]["content"])
    assert not any(item["type"] == "input_image" for item in npc_call["input"][0]["content"])
    assert any(item["type"] == "input_image" for item in agent_move_call["input"][0]["content"])
    assert any(item["type"] == "input_image" for item in agent_done_call["input"][0]["content"])
    _assert_no_named_product_terms(agent_ask_call["input"][0]["content"][0]["text"])
    _assert_no_named_product_terms(npc_call["input"][0]["content"][0]["text"])


def test_task2_agent_prompt_allows_clarification_at_any_step_based_on_image():
    module = _load_script_module()
    prompt = module.build_agent_task2_prompt(
        npc_initial_request="I want a red item with a shape I have in mind.",
        dialogue=[{"speaker": "npc", "utterance": "I want a red item with a shape I have in mind."}],
        visible_products=[],
        agent_decision_trace=[],
        navigation_trace=[],
    )

    assert "At any step" in prompt
    assert "current image" in prompt
    assert "ask_npc" in prompt
    assert "navigation_action" in prompt
    assert "Done" in prompt
    _assert_no_named_product_terms(prompt)


def test_task2_scenario_uses_shape_target_without_named_food_or_product_labels():
    module = _load_script_module()
    scenario = module.build_task2_scenario()

    assert scenario["hidden_target_product_id"] == "shape_red_cup_cylinder_001"
    assert "red products" in scenario["ambiguity"]
    target = next(product for product in scenario["catalog"] if product["product_id"] == "shape_red_cup_cylinder_001")
    assert target["target_shape"] == "cup-shaped cylinder with handle"
    assert "cup-shaped" in target["visual_description"]
    assert any(product["display_name_en"] == "red cup-shaped cylinder" for product in scenario["visible_products_for_agent"])
    assert any(product["display_name_en"] == "red round sphere A" for product in scenario["visible_products_for_agent"])
    assert any(product["display_name_en"] == "red round sphere B" for product in scenario["visible_products_for_agent"])
    assert all("visual_description" in product for product in scenario["visible_products_for_agent"])
    assert all("approx_location" in product for product in scenario["visible_products_for_agent"])
    assert all("object_type" not in product for product in scenario["visible_products_for_agent"])
    _assert_no_named_product_terms(json.dumps(scenario, ensure_ascii=False))


def test_task2_agent_prompt_discourages_repeated_moveahead_and_pickup_assumptions():
    module = _load_script_module()
    prompt = module.build_agent_task2_prompt(
        npc_initial_request="I want a red product.",
        dialogue=[
            {"speaker": "npc", "utterance": "I want a red product."},
            {"speaker": "agent", "utterance": "Which red product do you mean?"},
            {"speaker": "npc", "utterance": "The cup-shaped cylindrical one with a handle."},
        ],
        visible_products=[],
        agent_decision_trace=[],
        navigation_trace=[
            {"step_idx": 1, "action": "MoveAhead"},
            {"step_idx": 2, "action": "MoveAhead"},
        ],
    )

    assert "Do not repeat MoveAhead more than two consecutive navigation steps" in prompt
    assert "This is a find-and-confirm navigation task, not a pickup task" in prompt
    assert "If the clarified target is already clearly visible" in prompt
    _assert_no_named_product_terms(prompt)


def test_task2_customer_npc_run_writes_report_when_api_call_fails(tmp_path):
    module = _load_script_module()
    client = FailAfterClarificationClient()

    report = module.run_customer_npc_task(client=client, output_dir=tmp_path, width=480, height=320, max_steps=4)

    assert report["success"] is False
    assert report["termination"]["status"] == "error"
    assert report["termination"]["reason"] == "runtime_exception"
    assert "simulated API failure" in report["termination"]["error_message"]
    report_path = tmp_path / "task2_customer_npc_run.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["termination"]["status"] == "error"
    assert [turn["speaker"] for turn in saved["dialogue"]] == ["npc", "agent", "npc"]
