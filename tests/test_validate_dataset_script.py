import json
import sys
from importlib import util
from pathlib import Path


def load_validate_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "06_validate_dataset.py"
    sys.path.insert(0, str(script_path.parent))
    spec = util.spec_from_file_location("validate_dataset_script", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_episode(outputs_dir: Path) -> dict:
    image = outputs_dir / "images" / "episode_000001_step_000_rgb.png"
    meta = outputs_dir / "episodes" / "episode_000001_step_000_metadata.json"
    image.parent.mkdir(parents=True)
    meta.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    meta.write_text("{}", encoding="utf-8")

    return {
        "episode_id": "episode_000001",
        "schema_version": "0.1.0",
        "backend": "ai2thor",
        "scene": "FloorPlan1",
        "random_seed": 0,
        "capability_family": ["object_loco_navigation"],
        "task_type": "find_product",
        "prompt": "帮我找苹果",
        "target": {"target_product_id": "p_apple", "target_object_id": "Apple|1"},
        "episode_idx": 1,
        "max_steps": 20,
        "instruction": "帮我找苹果",
        "initial_state": {"agent_pose": {}, "held_object": None},
        "product_catalog": [
            {
                "product_id": "p_apple",
                "object_id": "Apple|1",
                "object_type": "Apple",
                "scene": "FloorPlan1",
                "shelf_id": "shelf_001",
                "display_name_zh": "苹果",
                "category": "fruit",
                "attributes": ["healthy"],
                "brand": "每日鲜果",
                "price": 5,
                "stock_status": "in_stock",
                "pickupable": True,
                "visible_at_start": True,
                "position": {"x": 1, "y": 1, "z": 1},
            }
        ],
        "shelf_graph": [{"shelf_id": "shelf_001", "source_object_id": "CounterTop|1"}],
        "npc_dialogue": [],
        "high_level_plan": [
            {
                "action": "look_at",
                "args": {"object_id": "Apple|1"},
                "preconditions": [],
                "expected_effects": [],
                "sim_actions": [{"action": "Teleport"}],
                "success": True,
                "failure_reason": None,
            }
        ],
        "sim_action_trace": [{"action": "Teleport", "success": True}],
        "observations": [
            {
                "step_idx": 0,
                "rgb_path": "images/episode_000001_step_000_rgb.png",
                "depth_path": None,
                "segmentation_path": None,
                "metadata_path": "episodes/episode_000001_step_000_metadata.json",
                "agent_pose": {},
                "held_object": None,
            }
        ],
        "metrics": {"target_found": True},
        "success": True,
        "failure_reason": None,
        "provenance": {"generator": "test"},
    }


def write_episode(tmp_path: Path, episode: dict) -> Path:
    episode_dir = tmp_path / "outputs" / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / "episode_000001.json"
    path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    return path


def test_validate_episode_file_accepts_complete_episode(tmp_path: Path):
    module = load_validate_module()
    episode = valid_episode(tmp_path / "outputs")
    path = write_episode(tmp_path, episode)

    module.validate_episode_file(path)


def test_validate_episode_file_rejects_missing_target_product(tmp_path: Path):
    module = load_validate_module()
    episode = valid_episode(tmp_path / "outputs")
    episode["target"]["target_product_id"] = "missing"
    path = write_episode(tmp_path, episode)

    try:
        module.validate_episode_file(path)
    except ValueError as exc:
        assert "target_product_id" in str(exc)
    else:
        raise AssertionError("expected missing target product to fail validation")


def test_validate_episode_file_rejects_untraceable_shelf(tmp_path: Path):
    module = load_validate_module()
    episode = valid_episode(tmp_path / "outputs")
    episode["shelf_graph"] = []
    path = write_episode(tmp_path, episode)

    try:
        module.validate_episode_file(path)
    except ValueError as exc:
        assert "shelf_id" in str(exc)
    else:
        raise AssertionError("expected untraceable shelf to fail validation")


def test_validate_episode_file_rejects_missing_action_args(tmp_path: Path):
    module = load_validate_module()
    episode = valid_episode(tmp_path / "outputs")
    episode["high_level_plan"][0]["args"] = {}
    path = write_episode(tmp_path, episode)

    try:
        module.validate_episode_file(path)
    except ValueError as exc:
        assert "look_at.object_id" in str(exc)
    else:
        raise AssertionError("expected missing action args to fail validation")


def test_episode_file_discovery_ignores_step_metadata_json(tmp_path: Path):
    module = load_validate_module()
    episode_dir = tmp_path / "outputs" / "episodes"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000001.json").write_text("{}", encoding="utf-8")
    (episode_dir / "episode_000001_step_000_metadata.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in module.iter_episode_files(episode_dir)] == ["episode_000001.json"]
