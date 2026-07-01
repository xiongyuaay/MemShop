from retail_thor.config import controller_config_from_scene_config, force_ai2thor_https_downloads


def test_controller_config_maps_custom_commit_key_to_ai2thor_argument():
    config = controller_config_from_scene_config(
        {
            "width": 800,
            "height": 600,
            "gridSize": 0.25,
            "macos_commit_id": "abc123",
        }
    )

    assert config["width"] == 800
    assert config["commit_id"] == "abc123"
    assert "macos_commit_id" not in config


def test_force_ai2thor_https_downloads_rewrites_public_s3_urls(monkeypatch):
    class FakeBuild:
        base_url = "http://s3-us-west-2.amazonaws.com/ai2-thor-public/"
        private_base_url = "http://s3-us-west-2.amazonaws.com/ai2-thor-private/"

    force_ai2thor_https_downloads(FakeBuild)

    assert FakeBuild.base_url.startswith("https://")
    assert FakeBuild.private_base_url.startswith("https://")
