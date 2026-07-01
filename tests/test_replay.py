import json
from pathlib import Path

from PIL import Image

from retail_thor.replay import replay_episode_file


def test_replay_episode_file_copies_observations_and_writes_report(tmp_path: Path):
    outputs = tmp_path / "outputs"
    image_dir = outputs / "images"
    episode_dir = outputs / "episodes"
    replay_dir = outputs / "replay"
    image_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_dir / "episode_000001_step_000_rgb.png")
    episode = {
        "episode_id": "episode_000001",
        "scene": "FloorPlan1",
        "random_seed": 0,
        "instruction": "帮我找苹果",
        "task_type": "find_product",
        "high_level_plan": [{"action": "look_at", "args": {"object_id": "Apple|1"}}],
        "observations": [
            {
                "step_idx": 0,
                "rgb_path": "images/episode_000001_step_000_rgb.png",
                "metadata_path": "episodes/episode_000001_step_000_metadata.json",
            }
        ],
        "provenance": {"observation_source": "demo_placeholder"},
    }
    episode_path = episode_dir / "episode_000001.json"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    report = replay_episode_file(episode_path, replay_dir)

    assert report["episode_id"] == "episode_000001"
    assert report["replay_exact"] is False
    assert report["steps"][0]["action"] == "look_at"
    assert Path(report["steps"][0]["rgb_replay_path"]).exists()
    assert (replay_dir / "episode_000001_replay_report.json").exists()
