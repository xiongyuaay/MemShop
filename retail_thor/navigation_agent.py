from __future__ import annotations

from typing import Any, Dict, Protocol


NAVIGATION_ACTIONS = (
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Done",
)

_ACTION_LOOKUP = {action.lower(): action for action in NAVIGATION_ACTIONS}


class StepBackend(Protocol):
    def step(self, action: Dict[str, Any]):
        ...


def normalize_navigation_action(action: str) -> str:
    normalized = _ACTION_LOOKUP.get(action.strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported navigation action: {action}")
    return normalized


class NavigationActionExecutor:
    def __init__(self, backend: StepBackend) -> None:
        self.backend = backend

    def execute(self, action: str, step_idx: int, thought: str = "") -> Dict[str, Any]:
        normalized = normalize_navigation_action(action)
        event = self.backend.step({"action": normalized})
        metadata = event.metadata
        return {
            "step_idx": step_idx,
            "action": normalized,
            "thought": thought,
            "success": metadata.get("lastActionSuccess", False),
            "error_message": metadata.get("errorMessage", ""),
            "agent_pose": metadata.get("agent", {}),
        }
