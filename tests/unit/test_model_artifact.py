"""
Tests for model artifact pinning.

The XGBoost classifier lives in Git LFS. A clone without ``git lfs pull`` holds
a ~130 byte pointer file instead of the 409,239 byte model, and the old code
happily tried to unpickle it. These tests assert the two guarantees the agent
depends on: a pointer is refused loudly, and the real artifact matches the
pinned SHA-256.
"""

import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from src.exceptions import ModelLoadError
from src.services.model_manager import LFS_POINTER_PREFIX, ModelManager
from tests.fakes import MockModel

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_model.py"

PINNED_ARTIFACT = config.MODEL_DIR / config.DEFAULT_MODEL_FILE


def _write_pointer(path: Path) -> Path:
    """Write a file that looks exactly like a Git-LFS pointer."""
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:523ff4dd301ac10855d64eb4a680e7ec0cd8985740ebd5679a9ed70f2a873c46\n"
        "size 409239\n"
    )
    return path


def _write_real_model(path: Path) -> Path:
    with open(path, "wb") as f:
        pickle.dump(MockModel(), f)
    return path


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return True
    with open(path, "rb") as f:
        return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


class TestVerifyModelArtifact:
    """The verification primitive itself."""

    def test_accepts_a_real_pickle(self, tmp_path):
        model_path = _write_real_model(tmp_path / "model.pkl")

        info = ModelManager.verify_model_artifact(model_path)

        expected_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert info["sha256"] == expected_sha
        assert info["size_bytes"] == model_path.stat().st_size
        assert info["path"] == str(model_path)

    def test_rejects_lfs_pointer(self, tmp_path):
        model_path = _write_pointer(tmp_path / "model.pkl")

        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.verify_model_artifact(model_path)

        assert "git-lfs pointer" in str(exc_info.value).lower()
        assert "git lfs pull" in str(exc_info.value).lower()

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.verify_model_artifact(tmp_path / "nope.pkl")

        assert "not found" in str(exc_info.value).lower()

    def test_rejects_size_mismatch(self, tmp_path):
        model_path = _write_real_model(tmp_path / "model.pkl")

        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.verify_model_artifact(model_path, expected_size_bytes=409239)

        assert "size mismatch" in str(exc_info.value).lower()

    def test_rejects_hash_mismatch(self, tmp_path):
        model_path = _write_real_model(tmp_path / "model.pkl")

        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.verify_model_artifact(model_path, expected_sha256="0" * 64)

        assert "sha-256 mismatch" in str(exc_info.value).lower()

    def test_accepts_matching_pin(self, tmp_path):
        model_path = _write_real_model(tmp_path / "model.pkl")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

        info = ModelManager.verify_model_artifact(
            model_path,
            expected_sha256=digest,
            expected_size_bytes=model_path.stat().st_size,
        )

        assert info["sha256"] == digest


class TestGetDefaultModelRefusesPointers:
    """``get_default_model`` fails loudly instead of unpickling a pointer."""

    def test_pointer_at_default_path_raises(self, tmp_path, monkeypatch):
        pointer = _write_pointer(tmp_path / config.DEFAULT_MODEL_FILE)
        monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", pointer)
        ModelManager.clear_cache()

        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.get_default_model()

        assert "git-lfs pointer" in str(exc_info.value).lower()

    def test_hash_pin_applies_to_the_pinned_artifact_name(self, tmp_path, monkeypatch):
        """A file with the pinned name but the wrong bytes is rejected."""
        impostor = _write_real_model(tmp_path / config.DEFAULT_MODEL_FILE)
        monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", impostor)
        ModelManager.clear_cache()

        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.get_default_model()

        assert "mismatch" in str(exc_info.value).lower()

    def test_injected_path_skips_the_hash_pin(self, tmp_path, monkeypatch):
        """Tests and custom deployments may inject any model file."""
        model_path = _write_real_model(tmp_path / "custom_model.pkl")
        monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", model_path)
        ModelManager.clear_cache()

        assert ModelManager.get_default_model() is not None


@pytest.mark.integration
class TestPinnedArtifactInTheWorkingCopy:
    """The artifact in this clone is the pinned one."""

    def test_working_copy_artifact_matches_the_pin(self):
        if _is_lfs_pointer(PINNED_ARTIFACT):
            pytest.skip(
                f"{PINNED_ARTIFACT.name} is a Git-LFS pointer; run `git lfs pull` "
                "or `python scripts/fetch_model.py` to verify the real artifact"
            )

        info = ModelManager.verify_model_artifact(
            PINNED_ARTIFACT,
            expected_sha256=config.DEFAULT_MODEL_SHA256,
            expected_size_bytes=config.DEFAULT_MODEL_SIZE_BYTES,
        )

        assert info["sha256"] == config.DEFAULT_MODEL_SHA256
        assert info["size_bytes"] == config.DEFAULT_MODEL_SIZE_BYTES

    def test_fetch_script_check_mode_passes_on_the_pinned_artifact(self):
        if _is_lfs_pointer(PINNED_ARTIFACT):
            pytest.skip(f"{PINNED_ARTIFACT.name} is a Git-LFS pointer; nothing to check")

        completed = subprocess.run(
            [sys.executable, str(FETCH_SCRIPT), "--check"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert config.DEFAULT_MODEL_SHA256 in completed.stdout

    def test_fetch_script_refuses_a_pointer(self, tmp_path):
        """`--check` on a pointer file exits non-zero with a clear message."""
        pointer = _write_pointer(tmp_path / config.DEFAULT_MODEL_FILE)

        completed = subprocess.run(
            [sys.executable, str(FETCH_SCRIPT), "--check", "--output", str(pointer)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        assert completed.returncode != 0
        assert "pointer" in (completed.stdout + completed.stderr).lower()
