from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from retail_thor.replay import load_episode, replay_episode_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "replay"))
    args = parser.parse_args()

    episode = load_episode(args.episode)
    print("Instruction:", episode["instruction"])
    for idx, step in enumerate(episode["high_level_plan"]):
        print(f"{idx:02d} {step['action']} {json.dumps(step['args'], ensure_ascii=False)}")
    report = replay_episode_file(args.episode, args.output_dir)
    print("Replay exact:", report["replay_exact"])
    print("Replay report:", report["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
