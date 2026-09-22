#!/usr/bin/env python3
"""
Fetch and verify the pinned XGBoost model artifact.

The model lives in Git LFS, so a plain clone holds a ~130 byte pointer instead
of the 409,239 byte classifier. This script is the escape hatch the Dokuprint
agent image uses at build time, and the check local devs run after `git lfs pull`:

    python scripts/fetch_model.py            # download + verify into models/
    python scripts/fetch_model.py --check    # verify the local file against the pin

The pin (URL, SHA-256, size) lives in ``src/config.py``. Nothing is written
unless the downloaded bytes match the pin, so a truncated download, a pointer
file or a swapped artifact fails loudly instead of quietly pricing documents
with the wrong model.

Exit codes: 0 verified, 1 verification failure, 2 download/IO failure.
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.services.model_manager import LFS_POINTER_PREFIX  # noqa: E402

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "print-cost-fetch-model/1.0"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    """True if the file is a Git-LFS pointer rather than the artifact."""
    with open(path, "rb") as f:
        return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def verify(path: Path, expected_sha256: str, expected_size_bytes: int) -> str:
    """Verify an artifact against the pin, or exit non-zero."""
    if not path.exists():
        print(f"FAIL: {path} does not exist", file=sys.stderr)
        raise SystemExit(1)

    if is_lfs_pointer(path):
        print(
            f"FAIL: {path} is a Git-LFS pointer, not the model "
            f"({path.stat().st_size} bytes). Run `git lfs pull` or "
            "`python scripts/fetch_model.py`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    size_bytes = path.stat().st_size
    if size_bytes != expected_size_bytes:
        print(
            f"FAIL: {path} is {size_bytes} bytes, expected {expected_size_bytes}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    sha256 = sha256_of(path)
    if sha256 != expected_sha256:
        print(
            f"FAIL: {path} has SHA-256 {sha256}, expected {expected_sha256}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"OK: {path}")
    print(f"    size   {size_bytes} bytes")
    print(f"    sha256 {sha256}")
    return sha256


def download(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` atomically, then return the path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fd, tmp_name = tempfile.mkstemp(dir=str(destination.parent), suffix=".part")
    tmp_path = Path(tmp_name)

    try:
        with urllib.request.urlopen(request, timeout=120) as response, os.fdopen(fd, "wb") as out:
            total = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
        print(f"    downloaded {total} bytes")
        os.replace(tmp_path, destination)
    except (urllib.error.URLError, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        print(f"FAIL: download error: {e}", file=sys.stderr)
        raise SystemExit(2)

    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEFAULT_MODEL_PATH,
        help=f"artifact path (default: {config.DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--url", default=config.DEFAULT_MODEL_URL, help="download URL"
    )
    parser.add_argument(
        "--sha256", default=config.DEFAULT_MODEL_SHA256, help="expected SHA-256"
    )
    parser.add_argument(
        "--size-bytes",
        type=int,
        default=config.DEFAULT_MODEL_SIZE_BYTES,
        help="expected size in bytes",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing file only; never download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the local file already matches the pin",
    )
    args = parser.parse_args(argv)

    if args.check:
        verify(args.output, args.sha256, args.size_bytes)
        return 0

    if args.output.exists() and not args.force:
        try:
            verify(args.output, args.sha256, args.size_bytes)
            print("Already pinned; nothing to do (use --force to re-download).")
            return 0
        except SystemExit:
            print("Local file does not match the pin; re-downloading.")

    download(args.url, args.output)
    verify(args.output, args.sha256, args.size_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
