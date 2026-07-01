import json
from pathlib import Path

from PIL import Image

from retail_thor.report import generate_demo_report, generate_ppt_assets


def make_episode(outputs: Path, idx: int, task_type: str) -> dict:
    image_dir = outputs / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image = image_dir / f"episode_{idx:06d}_step_000_rgb.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    return {
        "episode_id": f"episode_{idx:06d}",
        "task_type": task_type,
        "instruction": f"instruction {idx}",
        "target": {"target_product_id": f"product_{idx}"},
        "npc_dialogue": [{"speaker": "robot", "utterance": "hello"}] if "dialogue" in task_type else [],
        "high_level_plan": [{"action": "look_at"}, {"action": "finish"}],
        "observations": [{"rgb_path": str(image.relative_to(outputs))}],
        "metrics": {"target_found": True},
        "success": True,
    }


def test_generate_demo_report_and_ppt_assets(tmp_path: Path):
    outputs = tmp_path / "outputs"
    episodes = [
        make_episode(outputs, 0, "find_product"),
        make_episode(outputs, 1, "dialogue_find_or_substitute"),
        make_episode(outputs, 2, "pick_and_place"),
    ]
    report_path = outputs / "reports" / "demo_report.md"
    ppt_path = outputs / "reports" / "ppt_assets.md"

    generate_demo_report(episodes, report_path, outputs)
    generate_ppt_assets(report_path, ppt_path)

    report = report_path.read_text(encoding="utf-8")
    ppt = ppt_path.read_text(encoding="utf-8")
    assert "find_product" in report
    assert "dialogue_find_or_substitute" in report
    assert "pick_and_place" in report
    assert "episode_000000_step_000_rgb.png" in report
    assert "demo_report.md" in ppt
