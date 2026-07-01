import subprocess
import sys
from pathlib import Path


def test_generate_retail_shelf_screenshots_cli_exposes_real_ai2thor_help():
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/08_generate_retail_shelf_screenshots.py",
            "--help",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "real AI2-THOR single-shelf screenshot" in result.stdout
    assert "--output-dir" in result.stdout
