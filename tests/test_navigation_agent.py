import pytest

from retail_thor.navigation_agent import (
    NAVIGATION_ACTIONS,
    NavigationActionExecutor,
    normalize_navigation_action,
)


class FakeBackend:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return type(
            "FakeEvent",
            (),
            {
                "metadata": {
                    "lastActionSuccess": True,
                    "errorMessage": "",
                    "agent": {"position": {"x": 0, "y": 0, "z": 0}},
                }
            },
        )()


def test_navigation_actions_are_limited_to_safe_discrete_ai2thor_actions():
    assert NAVIGATION_ACTIONS == (
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Done",
    )
    assert normalize_navigation_action(" rotateleft ") == "RotateLeft"

    with pytest.raises(ValueError, match="unsupported navigation action"):
        normalize_navigation_action("PickupObject")


def test_navigation_executor_steps_backend_and_records_trace():
    backend = FakeBackend()
    executor = NavigationActionExecutor(backend)

    trace = executor.execute("MoveAhead", step_idx=3, thought="go closer")

    assert backend.actions == [{"action": "MoveAhead"}]
    assert trace == {
        "step_idx": 3,
        "action": "MoveAhead",
        "thought": "go closer",
        "success": True,
        "error_message": "",
        "agent_pose": {"position": {"x": 0, "y": 0, "z": 0}},
    }
