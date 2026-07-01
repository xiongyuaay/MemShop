from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict


def load_episode(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay_episode_file(path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    episode_path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = load_episode(episode_path)
    outputs_root = episode_path.parents[1]

    report = {
        "episode_id": episode["episode_id"],
        "scene": episode.get("scene"),
        "random_seed": episode.get("random_seed"),
        "instruction": episode.get("instruction"),
        "replay_exact": episode.get("provenance", {}).get("observation_source") != "demo_placeholder",
        "replay_note": None,
        "steps": [],
    }
    if not report["replay_exact"]:
        report["replay_note"] = "demo_placeholder observations are copied from generated data, not re-simulated"

    observations = {obs.get("step_idx"): obs for obs in episode.get("observations", [])}
    for idx, step in enumerate(episode.get("high_level_plan", [])):
        step_report = {
            "step_idx": idx,
            "action": step.get("action"),
            "args": step.get("args", {}),
            "rgb_replay_path": None,
        }
        obs = observations.get(idx)
        if obs and obs.get("rgb_path"):
            source = outputs_root / obs["rgb_path"]
            if source.exists():
                target = output_dir / f"{episode['episode_id']}_replay_step_{idx:03d}_rgb.png"
                shutil.copyfile(source, target)
                step_report["rgb_replay_path"] = str(target)
        report["steps"].append(step_report)

    report_path = output_dir / f"{episode['episode_id']}_replay_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
