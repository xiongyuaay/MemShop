from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

from retail_thor.report import generate_demo_report, generate_ppt_assets, load_episodes


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    outputs_root = ROOT / "outputs"
    episodes = load_episodes(outputs_root / "episodes")
    report_path = generate_demo_report(episodes, outputs_root / "reports" / "demo_report.md", outputs_root)
    ppt_path = generate_ppt_assets(report_path, outputs_root / "reports" / "ppt_assets.md")
    print(f"demo report: {report_path}")
    print(f"ppt assets: {ppt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
