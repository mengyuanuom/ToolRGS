"""Atomic downloads for deployment checkpoints published as release assets."""

import hashlib
from pathlib import Path
import urllib.request

from utils.pretrained import _download_ssl_context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_deployment_checkpoint(path, url="", sha256="") -> Path:
    target = Path(path).expanduser()
    expected = str(sha256 or "").strip().lower()
    if target.is_file() and target.stat().st_size:
        actual = _sha256(target) if expected else ""
        if not expected or actual == expected:
            return target
        if not url:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {target}")
        print(
            "[deployment] checkpoint is stale; downloading the configured "
            f"release asset\n  path: {target}\n  found: {actual}\n  want:  {expected}"
        )
    if not url:
        raise FileNotFoundError(
            f"Checkpoint does not exist: {target}. Set model.checkpoint_url "
            "to a published ToolRGS release asset or copy the file manually."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        str(url), headers={"User-Agent": "ToolRGS-deployment/1.0"}
    )
    print(f"[deployment] downloading checkpoint\n  from: {url}\n  to:   {target}")
    try:
        with urllib.request.urlopen(
            request, timeout=60, context=_download_ssl_context()
        ) as response:
            with temporary.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
        if expected and _sha256(temporary) != expected:
            raise RuntimeError("Downloaded checkpoint failed SHA-256 validation")
        # os.replace semantics keep the stale checkpoint intact until the new
        # download has passed SHA-256 validation, then swap it atomically.
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
