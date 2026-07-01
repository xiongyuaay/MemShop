from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASK_ORDER = ["find_product", "dialogue_find_or_substitute", "pick_and_place"]


def load_episodes(episode_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(episode_dir.glob("episode_*.json")) if "_step_" not in path.stem]


def generate_demo_report(episodes: list[dict[str, Any]], report_path: Path, outputs_root: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retail THOR Demo Report",
        "",
        "This report is generated from local episode JSON files.",
        "",
        "## Summary",
        "",
        f"- Episodes: {len(episodes)}",
        f"- Success: {sum(1 for episode in episodes if episode.get('success'))}",
        "",
    ]

    for task_type in TASK_ORDER:
        episode = next((item for item in episodes if item.get("task_type") == task_type), None)
        if not episode:
            continue
        lines.extend(_episode_section(episode, outputs_root))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_ppt_assets(report_path: Path, ppt_path: Path) -> Path:
    ppt_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# PPT Asset Index",
            "",
            f"- Demo report: {report_path.name}",
            "- Include one sample each for find_product, dialogue_find_or_substitute, and pick_and_place.",
            "- Use report image links as screenshot material.",
            "- Use action lists as the planning trace slide.",
            "- Use metrics blocks as the evaluation slide.",
            "",
        ]
    )
    ppt_path.write_text(content, encoding="utf-8")
    return ppt_path


def _episode_section(episode: dict[str, Any], outputs_root: Path) -> list[str]:
    observations = episode.get("observations", [])
    first_image = observations[0].get("rgb_path") if observations else None
    image_link = first_image or ""
    actions = ", ".join(step.get("action", "") for step in episode.get("high_level_plan", []))
    dialogue = " / ".join(turn.get("utterance", "") for turn in episode.get("npc_dialogue", []))
    metrics = json.dumps(episode.get("metrics", {}), ensure_ascii=False)

    lines = [
        f"## {episode.get('task_type')}",
        "",
        f"- Episode: `{episode.get('episode_id')}`",
        f"- Instruction: {episode.get('instruction')}",
        f"- Target: `{json.dumps(episode.get('target', {}), ensure_ascii=False)}`",
        f"- Success: `{episode.get('success')}`",
        f"- Actions: `{actions}`",
    ]
    if dialogue:
        lines.append(f"- NPC dialogue: {dialogue}")
    if image_link:
        lines.append(f"- Screenshot: `{image_link}`")
        lines.append("")
        lines.append(f"![{episode.get('episode_id')}]({image_link})")
    lines.extend(["", f"- Metrics: `{metrics}`", ""])
    return lines
