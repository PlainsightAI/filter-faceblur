"""
Tests for the shared verify_sha256 helper.

Used by both YuNetDetector._autodownload and DnnDetector._download to
optionally pin model artifacts. Unset (None / empty string) hashes are
a no-op (matching the existing trust model); when set, mismatches remove
the offending file and raise ValueError so the next download attempt is
clean.
"""

import hashlib

import pytest

from filter_faceblur.model.detectors.base_detector import verify_sha256


@pytest.fixture
def model_file(tmp_path):
    content = b"pretend model bytes"
    path = tmp_path / "model.bin"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def test_none_expected_is_noop(model_file):
    path, _ = model_file
    verify_sha256(path, None)
    assert path.exists()


def test_empty_expected_is_noop(model_file):
    path, _ = model_file
    verify_sha256(path, "")
    assert path.exists()


def test_matching_hash_passes(model_file):
    path, sha = model_file
    verify_sha256(path, sha)
    assert path.exists()


def test_matching_hash_is_case_insensitive(model_file):
    path, sha = model_file
    verify_sha256(path, sha.upper())
    assert path.exists()


def test_mismatch_raises_and_removes_file(model_file):
    path, _ = model_file
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64, label="my-model")
    # File removed so the next download attempt starts clean.
    assert not path.exists()


def test_mismatch_message_includes_label(model_file):
    path, _ = model_file
    with pytest.raises(ValueError, match="for my-model at"):
        verify_sha256(path, "0" * 64, label="my-model")


def test_handles_large_file_in_chunks(tmp_path):
    # Concatenate 200 KB of bytes — bigger than the 64KB read chunk in the
    # implementation, so multiple .update() calls happen.
    content = (b"abc" * 70_000)[:200_000]
    path = tmp_path / "big.bin"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    verify_sha256(path, expected)
    assert path.exists()
