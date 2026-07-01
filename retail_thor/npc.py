from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from retail_thor.catalog import choose_substitute
from retail_thor.world_knowledge_manager import WorldKnowledgeManager


@dataclass(frozen=True)
class NPCPlannerResponse:
    utterance: str
    rationale: str = ""
    product_id: str | None = None
    wkm_calls: list[dict[str, Any]] | None = None


class NPCPlanner(Protocol):
    model: str

    def respond(
        self,
        *,
        wkm: WorldKnowledgeManager,
        target_product_id: str,
        initial_request: str,
        agent_utterance: str,
        dialogue_history: list[dict[str, str]],
    ) -> NPCPlannerResponse:
        ...


class CustomerNPC:
    """Customer simulator; it only speaks and never outputs embodied actions."""

    def __init__(
        self,
        initial_request: str,
        target_product_id: str,
        wkm: WorldKnowledgeManager,
        planner: NPCPlanner | None = None,
    ) -> None:
        self.initial_request = initial_request
        self.target_product_id = target_product_id
        self.wkm = wkm
        self.planner = planner if planner is not None else GPT4oMiniNPCPlanner()
        self.dialogue_history: list[dict[str, str]] = []

    def initial_utterance(self) -> str:
        if not self.dialogue_history:
            self.dialogue_history.append({"speaker": "npc", "utterance": self.initial_request})
        return self.initial_request

    def respond(self, agent_utterance: str) -> NPCPlannerResponse:
        if not self.dialogue_history:
            self.initial_utterance()
        self.dialogue_history.append({"speaker": "agent", "utterance": agent_utterance})
        response = self.planner.respond(
            wkm=self.wkm,
            target_product_id=self.target_product_id,
            initial_request=self.initial_request,
            agent_utterance=agent_utterance,
            dialogue_history=self.dialogue_history,
        )
        self.dialogue_history.append({"speaker": "npc", "utterance": response.utterance})
        return response


class GPT4oMiniNPCPlanner:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        max_output_tokens: int = 160,
    ) -> None:
        self.model = model
        self.client = client if client is not None else self._default_client()
        self.max_output_tokens = max_output_tokens

    def respond(
        self,
        *,
        wkm: WorldKnowledgeManager,
        target_product_id: str,
        initial_request: str,
        agent_utterance: str,
        dialogue_history: list[dict[str, str]],
    ) -> NPCPlannerResponse:
        target_product = wkm.get_product(target_product_id)
        wkm_calls = wkm.consume_call_log()
        prompt = self._build_prompt(
            target_product=target_product,
            initial_request=initial_request,
            agent_utterance=agent_utterance,
            dialogue_history=dialogue_history,
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            text={"format": _npc_response_json_schema()},
            max_output_tokens=self.max_output_tokens,
        )
        parsed = _loads_json_object(_response_text(response))
        return NPCPlannerResponse(
            utterance=str(parsed.get("utterance", "")),
            rationale=str(parsed.get("rationale", "")),
            product_id=target_product_id,
            wkm_calls=wkm_calls,
        )

    @staticmethod
    def _default_client():
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI Python package is not installed. Run `pip install -r requirements.txt`.") from exc
        return OpenAI()

    @staticmethod
    def _build_prompt(
        *,
        target_product: dict[str, Any],
        initial_request: str,
        agent_utterance: str,
        dialogue_history: list[dict[str, str]],
    ) -> str:
        payload = {
            "role": "customer_npc",
            "initial_request": initial_request,
            "agent_utterance": agent_utterance,
            "dialogue_history": dialogue_history,
            "hidden_target_product_from_wkm": {
                "product_id": target_product.get("product_id"),
                "display_name_en": target_product.get("display_name_en", ""),
                "display_name_zh": target_product.get("display_name_zh", ""),
                "category": target_product.get("category", ""),
                "attributes": list(target_product.get("attributes", [])),
                "brand": target_product.get("brand", ""),
                "shelf_id": target_product.get("shelf_id", ""),
            },
        }
        return (
            "You are a customer NPC in a retail embodied-AI task. "
            "You are not the embodied agent and you must not control movement. "
            "Answer only as the customer, using the hidden target product information from WKM. "
            "If the embodied agent asks a clarification question, answer the requested attribute concisely. "
            "Do not output embodied actions, navigation actions, product IDs for the agent to execute, or plans. "
            "Return JSON only.\n"
            + json.dumps(payload, ensure_ascii=False)
        )


def resolve_dialogue_task(instruction: str, catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    if "酸奶" in instruction:
        substitute = choose_substitute(
            catalog,
            {"display_name_zh": "酸奶", "category": "dairy", "attributes": ["healthy"], "missing_reason": "not_in_scene"},
        )
        dialogue = [
            {
                "speaker": "npc",
                "utterance": instruction,
                "intent": "request_product",
                "state_delta": {"requested_product": "酸奶", "missing_reason": "not_in_scene"},
            },
            {
                "speaker": "agent",
                "utterance": "当前没有酸奶，我会找一个健康替代品。",
                "intent": "recommend_substitute",
                "state_delta": substitute or {},
            }
        ]
        return {
            "requires_clarification": False,
            "selected_product_id": substitute["substitute_product_id"] if substitute else None,
            "dialogue": dialogue,
        }

    if _is_ambiguous(instruction, catalog):
        return {
            "requires_clarification": True,
            "selected_product_id": None,
            "dialogue": [
                {
                    "speaker": "npc",
                    "utterance": instruction,
                    "intent": "request_product",
                    "state_delta": {},
                },
                {
                    "speaker": "agent",
                    "utterance": "你想找哪一类商品，或者有什么偏好的属性？",
                    "intent": "clarify_product_constraint",
                    "state_delta": {},
                }
            ],
        }

    return {
        "requires_clarification": False,
        "selected_product_id": catalog[0]["product_id"] if catalog else None,
        "dialogue": [],
    }


def _is_ambiguous(instruction: str, catalog: List[Dict[str, Any]]) -> bool:
    broad_terms = ("吃的", "东西", "商品")
    return any(term in instruction for term in broad_terms) and len(catalog) > 1


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


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("NPC planner response must be a JSON object")
    return data


def _npc_response_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "customer_npc_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "utterance": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["utterance", "rationale"],
        },
    }
