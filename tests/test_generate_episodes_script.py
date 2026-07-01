import json
import sys
from importlib import util
from pathlib import Path


def load_generate_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "04_generate_episodes.py"
    sys.path.insert(0, str(script_path.parent))
    spec = util.spec_from_file_location("generate_episodes_script", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_episodes_demo_mode_writes_json_images_and_metadata(tmp_path: Path):
    module = load_generate_module()
    templates = {
        "find_product": [{"instruction": "帮我找一个水果", "constraint": {"category": "fruit"}}],
        "dialogue_find_or_substitute": [{"instruction": "如果没有酸奶，帮我找一个健康食品"}],
        "pick_and_place": [
            {"instruction": "帮我把苹果放到购物栏", "constraint": {"category": "fruit"}, "placement": "cart"}
        ],
    }

    episodes = module.generate_episodes(
        catalog=[],
        templates=templates,
        out_dir=tmp_path / "outputs" / "episodes",
        outputs_root=tmp_path / "outputs",
        num_episodes=6,
        seed=1,
        task_types=["find_product", "dialogue_find_or_substitute", "pick_and_place"],
        demo_mode=True,
    )

    assert len(episodes) == 6
    assert {episode["task_type"] for episode in episodes} == {
        "find_product",
        "dialogue_find_or_substitute",
        "pick_and_place",
    }
    for episode in episodes:
        episode_path = tmp_path / "outputs" / "episodes" / f"{episode['episode_id']}.json"
        assert episode_path.exists()
        assert episode["observations"]
        for obs in episode["observations"]:
            assert (tmp_path / "outputs" / obs["rgb_path"]).exists()
            assert (tmp_path / "outputs" / obs["metadata_path"]).exists()
        saved = json.loads(episode_path.read_text(encoding="utf-8"))
        assert saved["provenance"]["observation_source"] == "demo_placeholder"


def test_cli_accepts_output_root_for_isolated_dataset(tmp_path: Path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "isolated_outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/04_generate_episodes.py",
            "--num-episodes",
            "1",
            "--task-types",
            "find_product",
            "--demo-mode",
            "--output-root",
            str(output_root),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "episodes" / "episode_000000.json").exists()
    assert (output_root / "images" / "episode_000000_step_000_rgb.png").exists()


def test_cli_accepts_custom_catalog_path(tmp_path: Path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "outputs"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "product_id": "apple",
                    "object_id": "Apple|custom",
                    "object_type": "Apple",
                    "scene": "FloorPlan1",
                    "shelf_id": "shelf_custom_001",
                    "source_receptacle_id": "CounterTop|custom",
                    "display_name_zh": "苹果",
                    "category": "fruit",
                    "attributes": ["healthy"],
                    "brand": "demo",
                    "price": 5,
                    "stock_status": "in_stock",
                    "pickupable": True,
                    "visible_at_start": True,
                    "position": {"x": 0, "y": 1, "z": 0},
                    "is_food": True,
                    "scene_randomization": {"randomized": True, "random_seed": 11, "random_spawn_success": True},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04_generate_episodes.py",
            "--catalog",
            str(catalog_path),
            "--num-episodes",
            "1",
            "--seed",
            "2",
            "--task-types",
            "find_product",
            "--demo-mode",
            "--output-root",
            str(output_root),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    episode = json.loads((output_root / "episodes" / "episode_000000.json").read_text(encoding="utf-8"))
    assert episode["target"]["target_object_id"] == "Apple|custom"
    assert episode["provenance"]["randomization"]["random_seed"] == 11
