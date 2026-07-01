from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from retail_thor.navigation_agent import NAVIGATION_ACTIONS, normalize_navigation_action


@dataclass(frozen=True)
class NavigationDecision:
    action: str
    rationale: str
    confidence: float


class OpenAIVLMNavigationBrain:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        client: Any | None = None,
        max_output_tokens: int = 160,
    ) -> None:
        self.model = model
        self.client = client if client is not None else self._default_client()
        self.max_output_tokens = max_output_tokens

    def decide_next_action(
        self,
        instruction: str,
        image_path: str | Path,
        navigation_history: List[Dict[str, Any]] | None = None,
        scene_context: Dict[str, Any] | None = None,
    ) -> NavigationDecision:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self._build_prompt(instruction, navigation_history, scene_context)},
                        {"type": "input_image", "image_url": _image_to_data_url(image_path)},
                    ],
                }
            ],
            text={"format": _navigation_json_schema()},
            max_output_tokens=self.max_output_tokens,
        )
        return parse_navigation_decision(_response_text(response))

    @staticmethod
    def _default_client():
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI Python package is not installed. Run `pip install -r requirements.txt`.") from exc
        return OpenAI()

    @staticmethod
    def _build_prompt(
        instruction: str,
        navigation_history: List[Dict[str, Any]] | None,
        scene_context: Dict[str, Any] | None,
    ) -> str:
        payload = {
            "instruction": instruction,
            "allowed_actions": list(NAVIGATION_ACTIONS),
            "navigation_history": navigation_history or [],
            "scene_context": scene_context or {},
        }
        return (
            "You are the navigation brain for an embodied agent in AI2-THOR. "
            "Choose exactly one next navigation action from allowed_actions. "
            "Use MoveAhead to approach, RotateLeft/RotateRight to turn, LookUp/LookDown to adjust camera pitch, "
            "and Done only when the navigation objective is achieved. "
            "Return JSON only.\n"
            + json.dumps(payload, ensure_ascii=False)
        )


def parse_navigation_decision(text: str) -> NavigationDecision:
    data = _loads_json_object(text)
    action = normalize_navigation_action(str(data.get("action", "")))
    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return NavigationDecision(
        action=action,
        rationale=str(data.get("rationale", "")),
        confidence=confidence,
    )


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise ValueError("OpenAI response does not contain output_text")


def _loads_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("navigation decision must be a JSON object")
    return data


def _image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _navigation_json_schema() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "navigation_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": list(NAVIGATION_ACTIONS)},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["action", "rationale", "confidence"],
        },
    }
