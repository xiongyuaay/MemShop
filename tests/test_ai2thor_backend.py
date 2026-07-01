from pathlib import Path

import numpy as np

from retail_thor.ai2thor_backend import AI2ThorBackend


class FakeEvent:
    def __init__(self):
        self.frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.depth_frame = np.ones((2, 2), dtype=np.float32)
        self.instance_segmentation_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.metadata = {
            "lastActionSuccess": True,
            "errorMessage": "",
            "agent": {"position": {"x": 0, "y": 0, "z": 0}},
            "heldObjectPose": None,
        }


class FakeController:
    def __init__(self, **config):
        self.config = config
        self.steps = []
        self.stopped = False

    def reset(self, scene):
        self.scene = scene
        return FakeEvent()

    def step(self, **action):
        self.steps.append(action)
        return FakeEvent()

    def stop(self):
        self.stopped = True


def test_backend_wraps_reset_step_observation_and_stop(tmp_path: Path):
    backend = AI2ThorBackend({"width": 800}, controller_factory=FakeController)

    backend.reset("FloorPlan1", seed=7)
    event = backend.step({"action": "Done"})
    observation = backend.save_observation(tmp_path, "episode_000001", 0, relative_to=tmp_path)
    backend.stop()

    assert event.metadata["lastActionSuccess"] is True
    assert backend.controller.config == {"width": 800}
    assert backend.controller.steps[0]["action"] == "InitialRandomSpawn"
    assert backend.controller.steps[0]["randomSeed"] == 7
    assert backend.controller.steps[1] == {"action": "Done"}
    assert observation["last_action_success"] is True
    assert observation["error_message"] == ""
    assert observation["rgb_path"] == "episode_000001_step_000_rgb.png"
    assert (tmp_path / "episode_000001_step_000_rgb.png").exists()
    assert (tmp_path / "episode_000001_step_000_depth.npy").exists()
    assert (tmp_path / "episode_000001_step_000_seg.png").exists()
    assert backend.controller.stopped is True
