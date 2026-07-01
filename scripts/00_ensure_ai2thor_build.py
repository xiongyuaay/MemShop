from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from retail_thor.build_cache import BuildSpec, ensure_ai2thor_build


ROOT = Path(__file__).resolve().parents[1]


def load_default_commit_id() -> str:
    scene_config = yaml.safe_load((ROOT / "configs" / "scenes.yaml").read_text(encoding="utf-8"))
    return scene_config["ai2thor"]["macos_commit_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and install the macOS AI2-THOR Unity build with resume support.")
    parser.add_argument("--platform", default="OSXIntel64")
    parser.add_argument("--commit-id", default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = BuildSpec(platform=args.platform, commit_id=args.commit_id or load_default_commit_id())
    report = ensure_ai2thor_build(spec, force_download=args.force_download)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
