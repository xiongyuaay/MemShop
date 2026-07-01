import json
from pathlib import Path

from PIL import Image

from retail_thor.vlm_navigation_brain import OpenAIVLMNavigationBrain


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
                        "action": "RotateRight",
                        "rationale": "The target shelf is on the right.",
                        "confidence": 0.82,
                    }
                )
            },
        )()


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_openai_vlm_navigation_brain_sends_image_and_returns_navigation_decision(tmp_path: Path):
    image_path = tmp_path / "obs.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)
    client = FakeOpenAIClient()
    brain = OpenAIVLMNavigationBrain(client=client, model="test-vlm")

    decision = brain.decide_next_action(
        instruction="走到货架前",
        image_path=image_path,
        navigation_history=[{"action": "MoveAhead", "success": True}],
        scene_context={"shelf_id": "shelf_FloorPlan1_010"},
    )

    assert decision.action == "RotateRight"
    assert decision.rationale == "The target shelf is on the right."
    assert decision.confidence == 0.82

    call = client.responses.calls[0]
    assert call["model"] == "test-vlm"
    assert call["text"]["format"]["schema"]["properties"]["action"]["enum"] == [
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Done",
    ]
    content = call["input"][0]["content"]
    assert any(item["type"] == "input_text" and "走到货架前" in item["text"] for item in content)
    image_items = [item for item in content if item["type"] == "input_image"]
    assert image_items
    assert image_items[0]["image_url"].startswith("data:image/png;base64,")
