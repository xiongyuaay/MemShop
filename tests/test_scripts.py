import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_script_entrypoints_can_import_project_package():
    result = subprocess.run(
        [sys.executable, "scripts/05_replay_episode.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_build_cache_script_entrypoint_can_import_project_package():
    result = subprocess.run(
        [sys.executable, "scripts/00_ensure_ai2thor_build.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_scan_scene_script_exposes_randomization_arguments():
    result = subprocess.run(
        [sys.executable, "scripts/02_scan_scene.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--randomize" in result.stdout
    assert "--seed" in result.stdout
    assert "--output-dir" in result.stdout


def test_build_catalog_script_exposes_inventory_and_output_arguments():
    result = subprocess.run(
        [sys.executable, "scripts/03_build_catalog.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--inventory-dir" in result.stdout
    assert "--output" in result.stdout


def test_vlm_navigation_script_exposes_navigation_arguments():
    result = subprocess.run(
        [sys.executable, "scripts/08_vlm_navigate.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--instruction" in result.stdout
    assert "--model" in result.stdout
    assert "--max-steps" in result.stdout
