from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from PIL import Image

from retail_thor.fbx_navigation_run import (
    NavigationCameraState,
    apply_navigation_action,
    build_navigation_prompt,
    render_navigation_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "retail_thor_demo"
SHELF_FBX = ROOT / "store-shelves" / "source" / "grocery_shelf.fbx"
SCRIPT = PROJECT / "scripts" / "10_run_gpt4o_fbx_shelf_navigation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_gpt4o_fbx_shelf_navigation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exposes_hardcoded_gpt4o_api_and_clash_proxy_config():
    module = _load_script_module()
    source = SCRIPT.read_text(encoding="utf-8")

    assert module.OPENAI_MODEL.startswith("gpt-4o")
    assert module.CLASH_HTTP_PROXY == "http://127.0.0.1:10808"
    assert module.CLASH_HTTPS_PROXY == "http://127.0.0.1:10808"
    assert isinstance(module.OPENAI_API_KEY, str)
    assert module.OPENAI_API_KEY.strip()
    assert "os.environ.get(\"OPENAI_API_KEY\"" not in source
    assert "report_example_without_live_api_call" not in source


def test_script_help_describes_real_gpt4o_run_and_outputs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run a real GPT-4o visual navigation loop" in result.stdout
    assert "--max-steps" in result.stdout
    assert "--output-dir" in result.stdout


def test_navigation_prompt_contains_allowed_actions_and_history():
    prompt = build_navigation_prompt(
        instruction="找到蓝色柱形商品，较细的。三个中任意一个都可以。",
        navigation_history=[{"step_idx": 0, "action": "MoveAhead", "success": True}],
        scene_context={"scene": "standalone_store_shelf_fbx"},
    )

    assert "gpt-4o" not in prompt.lower()
    assert "MoveAhead" in prompt
    assert "RotateLeft" in prompt
    assert "LookDown" in prompt
    assert "找到蓝色柱形商品" in prompt
    assert "\"navigation_history\"" in prompt
    assert "\"termination_policy\"" in prompt
    assert "at least one matching target instance" in prompt.lower()
    assert "near the image center" in prompt
    assert "Do not require pickup" in prompt


def test_navigation_observation_renderer_writes_distinct_running_screenshots(tmp_path):
    state = NavigationCameraState(position=(-190.0, -110.0, 78.0), yaw_degrees=30.0, pitch_degrees=-2.0)
    first = tmp_path / "step_000_input.png"
    second = tmp_path / "step_001_input.png"

    render_navigation_observation(SHELF_FBX, first, state, width=480, height=320)
    moved = apply_navigation_action(state, "MoveAhead")
    render_navigation_observation(SHELF_FBX, second, moved, width=480, height=320)

    assert first.exists()
    assert second.exists()
    assert Image.open(first).size == (480, 320)
    assert Image.open(second).size == (480, 320)
    assert first.read_bytes() != second.read_bytes()


def test_navigation_action_updates_camera_state_with_allowed_actions_only():
    state = NavigationCameraState(position=(-190.0, -110.0, 78.0), yaw_degrees=30.0, pitch_degrees=-2.0)

    moved = apply_navigation_action(state, "MoveAhead")
    turned = apply_navigation_action(state, "RotateRight")
    looked = apply_navigation_action(state, "LookDown")
    done = apply_navigation_action(state, "Done")

    assert moved.position != state.position
    assert turned.yaw_degrees > state.yaw_degrees
    assert looked.pitch_degrees < state.pitch_degrees
    assert done == state
