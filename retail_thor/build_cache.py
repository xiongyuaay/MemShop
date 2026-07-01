from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping


AI2THOR_PUBLIC_BUILDS_URL = "https://ai2-thor-public.s3.us-west-2.amazonaws.com/builds"


@dataclass(frozen=True)
class BuildSpec:
    platform: str
    commit_id: str

    @property
    def name(self) -> str:
        return build_name(self.platform, self.commit_id)

    @property
    def zip_filename(self) -> str:
        return f"{self.name}.zip"


def build_name(platform: str, commit_id: str) -> str:
    return f"thor-{platform}-{commit_id}"


def ai2thor_base_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".ai2thor"


def build_paths(spec: BuildSpec, home: Path | None = None) -> Dict[str, Path]:
    base_dir = ai2thor_base_dir(home)
    return {
        "base_dir": base_dir,
        "cache_dir": base_dir / "cache" / "downloads",
        "releases_dir": base_dir / "releases",
        "tmp_dir": base_dir / "tmp",
        "zip_path": base_dir / "cache" / "downloads" / spec.zip_filename,
        "release_dir": base_dir / "releases" / spec.name,
    }


def build_url(spec: BuildSpec, base_url: str = AI2THOR_PUBLIC_BUILDS_URL) -> str:
    return f"{base_url.rstrip('/')}/{spec.zip_filename}"


def sha256_url(spec: BuildSpec, base_url: str = AI2THOR_PUBLIC_BUILDS_URL) -> str:
    return f"{base_url.rstrip('/')}/{spec.name}.sha256"


def parse_sha256_text(text: str) -> str:
    digest = text.strip().split()[0]
    if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise ValueError(f"Invalid sha256 digest: {digest!r}")
    return digest.lower()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_ready(release_dir: Path) -> bool:
    app_dirs = list(release_dir.glob("*.app"))
    if not app_dirs:
        return False

    for app_dir in app_dirs:
        plist = app_dir / "Contents" / "Info.plist"
        macos_dir = app_dir / "Contents" / "MacOS"
        if plist.exists() and macos_dir.is_dir() and any(p.is_file() for p in macos_dir.iterdir()):
            return True
    return False


def proxy_disabled_env(base_env: Mapping[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def curl_common_args() -> list[str]:
    return ["curl", "-L", "--fail", "--noproxy", "*"]


def fetch_text_with_curl(url: str, timeout_seconds: int = 60) -> str:
    result = subprocess.run(
        [*curl_common_args(), "--silent", "--show-error", "--max-time", str(timeout_seconds), url],
        text=True,
        capture_output=True,
        check=True,
        env=proxy_disabled_env(),
    )
    return result.stdout


def download_with_curl(url: str, dest: Path, retries: int = 10) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            *curl_common_args(),
            "--retry",
            str(retries),
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(dest),
            url,
        ],
        check=True,
        env=proxy_disabled_env(),
    )


def ensure_executable_bits(release_dir: Path) -> None:
    for executable in release_dir.glob("*.app/Contents/MacOS/*"):
        if executable.is_file():
            executable.chmod(executable.stat().st_mode | 0o755)


def install_zip(zip_path: Path, release_dir: Path, tmp_dir: Path) -> None:
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if release_ready(release_dir):
        ensure_executable_bits(release_dir)
        return

    extract_dir = Path(tempfile.mkdtemp(prefix=f"{release_dir.name}.", dir=tmp_dir))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        if release_dir.exists():
            shutil.rmtree(release_dir)
        shutil.move(str(extract_dir), str(release_dir))
        ensure_executable_bits(release_dir)
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


def ensure_ai2thor_build(
    spec: BuildSpec,
    home: Path | None = None,
    base_url: str = AI2THOR_PUBLIC_BUILDS_URL,
    force_download: bool = False,
) -> Dict[str, str | bool]:
    paths = build_paths(spec, home)
    release_dir = paths["release_dir"]
    zip_path = paths["zip_path"]

    if release_ready(release_dir):
        ensure_executable_bits(release_dir)
        return {
            "success": True,
            "build_name": spec.name,
            "release_dir": str(release_dir),
            "downloaded": False,
            "installed": False,
        }

    expected_sha = parse_sha256_text(fetch_text_with_curl(sha256_url(spec, base_url)))
    if force_download and zip_path.exists():
        zip_path.unlink()

    downloaded = False
    resume_from_bytes = zip_path.stat().st_size if zip_path.exists() else 0
    download_with_curl(build_url(spec, base_url), zip_path)
    downloaded = True

    actual_sha = sha256_file(zip_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Downloaded build digest mismatch: expected {expected_sha}, got {actual_sha}")

    install_zip(zip_path, release_dir, paths["tmp_dir"])
    if not release_ready(release_dir):
        raise RuntimeError(f"AI2-THOR release did not install correctly: {release_dir}")

    return {
        "success": True,
        "build_name": spec.name,
        "release_dir": str(release_dir),
        "zip_path": str(zip_path),
        "downloaded": downloaded,
        "resume_from_bytes": str(resume_from_bytes),
        "installed": True,
    }
