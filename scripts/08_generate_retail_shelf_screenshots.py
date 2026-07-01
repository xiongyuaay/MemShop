from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from retail_thor.real_shelf_render import render_real_single_shelf


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a real AI2-THOR single-shelf screenshot for reports.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "report_shelf_scenes"))
    args = parser.parse_args()

    manifest = render_real_single_shelf(Path(args.output_dir))
    print(f"generated real AI2-THOR shelf screenshots: {len(manifest['screenshots'])}")
    print(f"output directory: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
