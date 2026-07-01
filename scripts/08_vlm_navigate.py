from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from retail_thor.ai2thor_backend import AI2ThorBackend
from retail_thor.config import controller_config_from_scene_config
from retail_thor.navigation_agent import NavigationActionExecutor
from retail_thor.vlm_navigation_brain import OpenAIVLMNavigationBrain


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="导航到货架前，并在看清货架商品后停止。")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "vlm_navigation_demo"))
    args = parser.parse_args()

    scene_config = yaml.safe_load((ROOT / "configs" / "scenes.yaml").read_text(encoding="utf-8"))
    single_shelf = yaml.safe_load((ROOT / "configs" / "single_shelf.yaml").read_text(encoding="utf-8"))
    scene = args.scene or single_shelf.get("scene", "FloorPlan1")
    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    backend = AI2ThorBackend(controller_config_from_scene_config(scene_config["ai2thor"]))
    executor = NavigationActionExecutor(backend)
    brain = OpenAIVLMNavigationBrain(model=args.model)
    trace = []

    try:
        backend.reset(scene)
        for step_idx in range(args.max_steps):
            observation = backend.save_observation(image_dir, "vlm_nav", step_idx, relative_to=output_root)
            image_path = output_root / observation["rgb_path"]
            decision = brain.decide_next_action(
                instruction=args.instruction,
                image_path=image_path,
                navigation_history=trace,
                scene_context={
                    "scene": scene,
                    "target_shelf_id": single_shelf.get("shelf_id"),
                    "target_receptacle_id": single_shelf.get("source_object_id"),
                },
            )
            action_trace = executor.execute(decision.action, step_idx=step_idx, thought=decision.rationale)
            action_trace["confidence"] = decision.confidence
            action_trace["observation"] = observation
            trace.append(action_trace)
            if decision.action == "Done":
                break
    finally:
        backend.stop()

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "vlm_navigation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "instruction": args.instruction,
                "scene": scene,
                "model": args.model,
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"steps": len(trace), "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
