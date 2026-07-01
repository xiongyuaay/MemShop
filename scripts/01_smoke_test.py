from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from PIL import Image

from retail_thor.config import controller_config_from_scene_config, force_ai2thor_https_downloads


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
IMAGE_DIR = ROOT / "outputs" / "images"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    report = runtime_report(success=False)

    try:
        from ai2thor.controller import Controller

        force_ai2thor_https_downloads()
        config = controller_config_from_scene_config(
            yaml.safe_load((ROOT / "configs" / "scenes.yaml").read_text(encoding="utf-8"))["ai2thor"]
        )
        controller = Controller(scene="FloorPlan1", **config)
        event = controller.step(action="RotateRight")

        objects = event.metadata.get("objects", [])
        Image.fromarray(event.frame).save(IMAGE_DIR / "smoke_test.png")

        report.update(
            {
                "success": True,
                "scene": "FloorPlan1",
                "metadata_keys": sorted(event.metadata.keys()),
                "num_objects": len(objects),
                "num_visible": sum(1 for obj in objects if obj.get("visible")),
                "num_pickupable": sum(1 for obj in objects if obj.get("pickupable")),
                "num_receptacles": sum(1 for obj in objects if obj.get("receptacle")),
                "last_action_success": event.metadata.get("lastActionSuccess"),
                "screenshot": "outputs/images/smoke_test.png",
            }
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        controller.stop()
        return 0
    except Exception as exc:
        report["error"] = repr(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    finally:
        (REPORT_DIR / "runtime_env.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def runtime_report(success: bool) -> dict:
    return {
        "success": success,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "has_display_env": bool(platform.system() == "Darwin"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
