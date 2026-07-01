import subprocess
import sys
from pathlib import Path


def test_generate_fbx_shelf_screenshot_cli_exposes_standalone_fbx_help():
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/09_generate_fbx_shelf_screenshot.py",
            "--help",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "standalone FBX store-shelf screenshot" in result.stdout
    assert "--shelf-fbx" in result.stdout
    assert "--product-count" in result.stdout
