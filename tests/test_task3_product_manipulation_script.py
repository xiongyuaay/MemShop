from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "12_run_gpt4o_mini_fbx_product_manipulation_task.py"


class QueueResponses:
    def __init__(self):
        self.calls = []
        self.outputs = [
            {
                "action": "MoveAhead",
                "question_text": "",
                "rationale": "Move closer to the shelf before manipulating the red mug.",
                "confidence": 0.72,
            },
            {
                "action": "MoveAhead",
                "question_text": "",
                "rationale": "Continue approaching because pickup requires being near the target object.",
                "confidence": 0.74,
            },
            {
                "action": "MoveAhead",
                "question_text": "",
                "rationale": "Continue approaching the actual rendered product instance.",
                "confidence": 0.76,
            },
            {
                "action": "MoveAhead",
                "question_text": "",
                "rationale": "Move into pickup range of the rendered red mug.",
                "confidence": 0.78,
            },
            {
                "action": "MoveAhead",
                "question_text": "",
                "rationale": "Stop close enough to the rendered product instance for pickup.",
                "confidence": 0.8,
            },
            {
                "action": "PickupObject",
                "question_text": "",
                "rationale": "The red mug is visible and the agent is now close enough to pick it up.",
                "confidence": 0.82,
            },
            {
                "action": "RotateLeft",
                "question_text": "",
                "rationale": "Turn toward the destination shelf position.",
                "confidence": 0.7,
            },
            {
                "action": "DropObject",
                "question_text": "",
                "rationale": "The agent is holding the red mug and should place it onto the destination shelf position.",
                "confidence": 0.86,
            },
            {
                "action": "Done",
                "question_text": "",
                "rationale": "The red mug has been moved to the destination shelf position.",
                "confidence": 0.94,
            },
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.outputs.pop(0)
        return type("FakeResponse", (), {"output_text": json.dumps(payload)})()


class QueueClient:
    def __init__(self):
        self.responses = QueueResponses()


class AskQuestionResponses:
    def __init__(self):
        self.calls = []
        self.outputs = [
            {
                "action": "AskQuestion",
                "question_text": "Which shelf position should I move it to?",
                "rationale": "The destination is ambiguous.",
                "confidence": 0.62,
            },
            {
                "action": "Done",
                "question_text": "",
                "rationale": "Stopping after the question for this short test.",
                "confidence": 0.4,
            },
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.outputs.pop(0)
        return type("FakeResponse", (), {"output_text": json.dumps(payload)})()


class AskQuestionClient:
    def __init__(self):
        self.responses = AskQuestionResponses()


def _load_script_module():
    spec = importlib.util.spec_from_file_location("task3_product_manipulation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task3_script_uses_gpt4o_mini_and_expanded_action_space():
    module = _load_script_module()

    assert module.OPENAI_MODEL == "gpt-4o-mini"
    assert module.MANIPULATION_ACTIONS == (
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Done",
        "AskQuestion",
        "PickupObject",
        "DropObject",
    )
    schema = module._manipulation_json_schema()
    assert schema["schema"]["properties"]["action"]["enum"] == list(module.MANIPULATION_ACTIONS)


def test_task3_run_moves_actual_rendered_product_between_shelf_positions_and_requires_done(tmp_path):
    module = _load_script_module()
    client = QueueClient()

    report = module.run_product_manipulation_task(
        client=client,
        output_dir=tmp_path,
        width=480,
        height=320,
        max_steps=10,
    )

    assert report["run_type"] == "real_gpt4o_mini_product_manipulation_task"
    assert report["model"] == "gpt-4o-mini"
    assert report["instruction"] == "Move the red mug from its source shelf position to the destination shelf position."
    assert report["success"] is True
    assert report["termination"] == {
        "status": "done",
        "reason": "target_object_at_destination_shelf_slot",
        "done_action_required": True,
    }
    assert report["final_object_state"]["target_location"] == "destination_shelf_slot"
    assert report["final_object_state"]["held_object_id"] is None
    assert report["scenario"]["target_object"]["object_id"] == "Mug|product_index_003"
    assert report["scenario"]["target_object"]["renderer_product_index"] == 3
    assert report["scenario"]["target_object"]["object_type"] == "Mug"
    assert "world_center" in report["scenario"]["target_object"]
    assert [step["executed_action"] for step in report["trajectory"]] == [
        "MoveAhead",
        "MoveAhead",
        "MoveAhead",
        "MoveAhead",
        "MoveAhead",
        "PickupObject",
        "RotateLeft",
        "DropObject",
        "Done",
    ]
    assert [step["success"] for step in report["operation_trace"] if step["action"] in {"PickupObject", "DropObject"}] == [True, True]
    assert (tmp_path / "images" / "step_000_input.png").exists()
    assert report["operation_trace"][0]["distance_to_target"] <= report["scenario"]["pickup_distance_threshold"]
    assert (tmp_path / "images" / "step_008_input.png").exists()
    assert (tmp_path / "task3_product_manipulation_run.json").exists()
    assert (tmp_path / "raw_responses" / "step_000_response.json").exists()
    assert (tmp_path / "raw_responses" / "step_004_response.json").exists()
    assert all(any(item["type"] == "input_image" for item in call["input"][0]["content"]) for call in client.responses.calls)


def test_task3_prompt_describes_pickup_drop_done_and_ask_question_contract():
    module = _load_script_module()
    prompt = module.build_product_manipulation_prompt(
        instruction="Move the red mug from its source shelf position to the destination shelf position.",
        scene_context=module.build_task3_scenario(),
        object_state={"target_location": "source_shelf_slot", "held_object_id": None},
        history=[],
    )

    assert "AskQuestion(question_text)" in prompt
    assert "PickupObject" in prompt
    assert "DropObject" in prompt
    assert "Done only after" in prompt
    assert "target_location is destination_shelf_slot" in prompt
    assert "environment returns success=false" in prompt
    assert "target_too_far_for_pickup" in prompt


def test_task3_pickup_failure_is_environment_feedback_when_agent_is_too_far():
    module = _load_script_module()
    scenario = module.build_task3_scenario()
    object_state = {
        "target_object_id": scenario["target_object"]["object_id"],
        "target_location": "source_shelf_slot",
        "held_object_id": None,
    }
    far_camera_state = module.default_initial_navigation_state(module.DEFAULT_SHELF_FBX)

    result = module._apply_manipulation_action("PickupObject", object_state, scenario, far_camera_state)

    assert result["success"] is False
    assert result["message"] == "target_too_far_for_pickup"
    assert result["distance_to_target"] > scenario["pickup_distance_threshold"]
    assert result["environment_feedback"] == "PickupObject failed: agent is too far from the target object."
    assert object_state["target_location"] == "source_shelf_slot"
    assert object_state["held_object_id"] is None


def test_task3_script_does_not_draw_source_or_destination_boxes():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"SOURCE"' not in source
    assert '"DESTINATION"' not in source
    assert "draw.rectangle(source_box" not in source
    assert "draw.rectangle(destination_box" not in source


def test_task3_ask_question_action_records_question_without_state_change(tmp_path):
    module = _load_script_module()
    client = AskQuestionClient()

    report = module.run_product_manipulation_task(
        client=client,
        output_dir=tmp_path,
        width=480,
        height=320,
        max_steps=2,
    )

    assert report["question_trace"] == [
        {
            "step_idx": 0,
            "question_text": "Which shelf position should I move it to?",
            "rationale": "The destination is ambiguous.",
        }
    ]
    assert report["trajectory"][0]["executed_action"] == "AskQuestion"
    assert report["trajectory"][0]["state_before"]["target_location"] == "source_shelf_slot"
    assert report["trajectory"][0]["state_after"]["target_location"] == "source_shelf_slot"
    assert report["success"] is False
