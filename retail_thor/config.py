from __future__ import annotations

from typing import Any, Dict


def controller_config_from_scene_config(scene_config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(scene_config)
    macos_commit_id = config.pop("macos_commit_id", None)
    if macos_commit_id:
        config["commit_id"] = macos_commit_id
    return config


def force_ai2thor_https_downloads(build_module=None) -> None:
    if build_module is None:
        import ai2thor.build as build_module

    for attr in ("base_url", "private_base_url"):
        value = getattr(build_module, attr, "")
        if value.startswith("http://s3-us-west-2.amazonaws.com/"):
            setattr(build_module, attr, value.replace("http://", "https://", 1))
