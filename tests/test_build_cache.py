from pathlib import Path

from retail_thor.build_cache import (
    BuildSpec,
    build_name,
    build_paths,
    curl_common_args,
    download_with_curl,
    parse_sha256_text,
    proxy_disabled_env,
    release_ready,
)


def test_build_spec_uses_ai2thor_build_name_format():
    spec = BuildSpec(platform="OSXIntel64", commit_id="abc123")

    assert spec.name == "thor-OSXIntel64-abc123"
    assert spec.zip_filename == "thor-OSXIntel64-abc123.zip"
    assert build_name("Linux64", "def456") == "thor-Linux64-def456"


def test_build_paths_match_ai2thor_default_cache_layout(tmp_path: Path):
    paths = build_paths(BuildSpec(platform="OSXIntel64", commit_id="abc123"), home=tmp_path)

    assert paths["release_dir"] == tmp_path / ".ai2thor" / "releases" / "thor-OSXIntel64-abc123"
    assert paths["zip_path"] == tmp_path / ".ai2thor" / "cache" / "downloads" / "thor-OSXIntel64-abc123.zip"


def test_parse_sha256_text_accepts_digest_with_filename():
    digest = "a" * 64

    assert parse_sha256_text(f"{digest}  thor.zip\n") == digest


def test_release_ready_requires_macos_app_with_executable(tmp_path: Path):
    release_dir = tmp_path / "thor-OSXIntel64-abc123"
    macos_dir = release_dir / "thor-OSXIntel64-abc123.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    (macos_dir.parent / "Info.plist").write_text("<plist />", encoding="utf-8")

    assert not release_ready(release_dir)

    (macos_dir / "thor").write_text("", encoding="utf-8")

    assert release_ready(release_dir)


def test_curl_downloads_bypass_proxy_environment():
    env = proxy_disabled_env(
        {
            "HTTPS_PROXY": "http://127.0.0.1:7890",
            "http_proxy": "http://127.0.0.1:7890",
            "PATH": "/usr/bin",
        }
    )

    assert "HTTPS_PROXY" not in env
    assert "http_proxy" not in env
    assert env["NO_PROXY"] == "*"
    assert "--noproxy" in curl_common_args()


def test_download_uses_resume_without_predeleting_partial_file(tmp_path: Path, monkeypatch):
    partial = tmp_path / "build.zip"
    partial.write_bytes(b"partial")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("retail_thor.build_cache.subprocess.run", fake_run)

    download_with_curl("https://example.com/build.zip", partial)

    assert partial.read_bytes() == b"partial"
    assert "--continue-at" in calls[0][0]
    assert calls[0][1]["env"]["NO_PROXY"] == "*"
