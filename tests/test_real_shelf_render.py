import json
from pathlib import Path

import numpy as np
from PIL import Image

from retail_thor.real_shelf_render import render_real_single_shelf


class FakeEvent:
    def __init__(self, objects=None, success=True):
        self.frame = np.full((12, 16, 3), 120, dtype=np.uint8)
        self.metadata = {
            "lastActionSuccess": success,
            "errorMessage": "",
            "agent": {"position": {"x": -1.25, "y": 0.9, "z": 0}, "rotation": {"y": 90}},
            "objects": objects or [],
        }


class FakeController:
    def __init__(self, scene, **config):
        self.scene = scene
        self.config = config
        self.steps = []
        self.stopped = False
        self.objects = [
            {
                "objectId": "Shelf|+01.75|+00.55|-02.56",
                "objectType": "Shelf",
                "receptacle": True,
                "position": {"x": 1.75, "y": 0.55, "z": -2.56},
                "axisAlignedBoundingBox": {"center": {"x": 1.75, "y": 0.55, "z": -2.56}},
            },
            {
                "objectId": "Shelf|+01.75|+00.88|-02.56",
                "objectType": "Shelf",
                "receptacle": True,
                "position": {"x": 1.75, "y": 0.88, "z": -2.56},
                "axisAlignedBoundingBox": {"center": {"x": 1.75, "y": 0.88, "z": -2.56}},
            },
            {
                "objectId": "Bottle|1",
                "objectType": "Bottle",
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["Shelf|+01.75|+00.88|-02.56"],
            },
            {
                "objectId": "SoapBottle|1",
                "objectType": "SoapBottle",
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["Shelf|+01.75|+00.55|-02.56"],
            },
            {
                "objectId": "Egg|1",
                "objectType": "Egg",
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["Shelf|+01.75|+00.88|-02.56"],
            },
            {
                "objectId": "Vase|1",
                "objectType": "Vase",
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["Shelf|+01.75|+00.55|-02.56"],
            },
            {
                "objectId": "Statue|1",
                "objectType": "Statue",
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["Shelf|+01.75|+00.17|-02.56"],
            },
            {
                "objectId": "Fork|1",
                "objectType": "Fork",
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["CounterTop|-00.08|+01.15|00.00"],
            },
        ]

    def step(self, **action):
        self.steps.append(action)
        if action["action"] == "Teleport":
            return FakeEvent(self.objects, success=True)
        return FakeEvent(self.objects, success=True)

    def stop(self):
        self.stopped = True


def test_render_real_single_shelf_writes_real_ai2thor_manifest_and_frame(tmp_path: Path):
    manifest = render_real_single_shelf(
        output_dir=tmp_path,
        controller_factory=FakeController,
        controller_config={"width": 16, "height": 12},
    )

    assert manifest["source"] == "ai2thor_real_render"
    assert manifest["scene"] == "FloorPlan1"
    assert manifest["shelf"]["source_object_type"] == "Shelf"
    assert "Shelf|+01.75|+00.55|-02.56" in manifest["shelf"]["source_object_ids"]
    assert manifest["shelf_randomization"]["action"] == "InitialRandomSpawn"
    assert manifest["shelf_randomization"]["randomSeed"] == 86
    assert manifest["excluded_object_types"] == ["Statue"]
    assert manifest["rendered_object_types"] == ["Bottle", "Egg", "SoapBottle", "Vase"]
    assert "Statue" not in manifest["rendered_object_types"]
    assert manifest["view_policy"] == "cropped_single_shelf_focus"
    assert manifest["screenshots"][0]["path"] == "real_single_shelf.png"
    assert (tmp_path / "real_single_shelf.png").exists()
    assert (tmp_path / "real_single_shelf_raw.png").exists()
    assert (tmp_path / "real_single_shelf_metadata.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["source"] == "ai2thor_real_render"

    with Image.open(tmp_path / "real_single_shelf.png") as image:
        assert image.size == (16, 12)


def test_render_real_single_shelf_teleports_to_probe_pose(tmp_path: Path):
    controller_ref = {}

    class RecordingController(FakeController):
        def __init__(self, scene, **config):
            super().__init__(scene, **config)
            controller_ref["controller"] = self

    render_real_single_shelf(
        output_dir=tmp_path,
        controller_factory=RecordingController,
        controller_config={"width": 16, "height": 12},
    )

    actions = controller_ref["controller"].steps
    assert actions[1] == {
        "action": "InitialRandomSpawn",
        "randomSeed": 86,
        "forceVisible": True,
        "numPlacementAttempts": 5,
    }
    assert any(action["action"] == "Teleport" for action in actions)
    assert actions[-1]["position"] == {"x": 1.75, "y": 0.900999128818512, "z": -1.25}
    assert actions[-1]["rotation"]["y"] == 179.9557435
    assert actions[-1]["horizon"] == 10
    assert actions[-1]["action"] == "Teleport"
    assert controller_ref["controller"].stopped is True
