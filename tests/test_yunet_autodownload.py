"""Tests for YuNetDetector._autodownload and _download_model_jfrog.

Covers the rewrite that:
  - added the missing `return` after the OpenCV download path (the standard
    branch used to fall through into the JFrog branch, double-downloading the
    model to a hardcoded `filter_gaf/measurements/...` path).
  - replaced `subprocess.run(..., shell=True)` with an argv list, removing
    the command-injection vector through interpolated credentials.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from filter_faceblur.model.detectors.yunet_detector import YuNetDetector


@pytest.fixture
def detector():
    # Bypass __init__ so the cv2 model load doesn't run; we only exercise
    # the download helpers.
    instance = object.__new__(YuNetDetector)
    instance.api_key = None
    instance.username = None
    return instance


GITHUB_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
JFROG_URL = (
    "https://example.jfrog.io/artifactory/repo/face_detection_yunet_2023mar.onnx"
)


class TestAutoDownload:
    def test_returns_existing_path_without_download(self, detector, monkeypatch):
        monkeypatch.setattr(Path, "is_file", lambda self: True)
        with patch.object(detector, "_download_model_opencv") as opencv, \
             patch.object(detector, "_download_model_jfrog") as jfrog:
            result = detector._autodownload(GITHUB_URL)
        opencv.assert_not_called()
        jfrog.assert_not_called()
        assert isinstance(result, str)
        assert result.endswith("face_detection_yunet_2023mar.onnx")

    def test_github_url_no_creds_routes_to_opencv(self, detector, monkeypatch):
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv") as opencv, \
             patch.object(detector, "_download_model_jfrog") as jfrog:
            detector._autodownload(GITHUB_URL)
        opencv.assert_called_once()
        jfrog.assert_not_called()

    def test_github_url_with_creds_still_routes_to_opencv(self, detector, monkeypatch):
        # Creds alone don't trigger the JFrog path — URL must also be jfrog.
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv") as opencv, \
             patch.object(detector, "_download_model_jfrog") as jfrog:
            detector._autodownload(GITHUB_URL, api_key="k", username="u")
        opencv.assert_called_once()
        jfrog.assert_not_called()

    def test_jfrog_url_with_creds_routes_to_jfrog(self, detector, monkeypatch):
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv") as opencv, \
             patch.object(detector, "_download_model_jfrog") as jfrog:
            detector._autodownload(JFROG_URL, api_key="k", username="u")
        jfrog.assert_called_once()
        opencv.assert_not_called()

    @pytest.mark.parametrize("api_key,username", [
        (None, "u"),
        ("k", None),
        (None, None),
        ("", "u"),
        ("k", ""),
    ])
    def test_jfrog_url_missing_creds_falls_back_to_opencv(
        self, detector, monkeypatch, api_key, username
    ):
        # Missing or empty creds with a jfrog URL must NOT invoke curl with
        # `-u None:None` (the old behavior, which "worked" only because
        # github raw ignores junk auth headers).
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv") as opencv, \
             patch.object(detector, "_download_model_jfrog") as jfrog:
            detector._autodownload(JFROG_URL, api_key=api_key, username=username)
        opencv.assert_called_once()
        jfrog.assert_not_called()

    def test_opencv_path_does_not_fall_through_to_jfrog(self, detector, monkeypatch):
        # Regression: pre-rewrite, execution kept running after
        # _download_model_opencv and re-downloaded via curl into
        # filter_gaf/measurements/model/weights/.
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv"), \
             patch.object(detector, "_download_model_jfrog") as jfrog, \
             patch("subprocess.run") as run:
            detector._autodownload(GITHUB_URL)
        jfrog.assert_not_called()
        run.assert_not_called()

    def test_return_value_is_str(self, detector, monkeypatch):
        # cv2.FaceDetectorYN_create takes a str path; returning a PosixPath
        # works on Linux but is risky across cv2 builds.
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv"):
            result = detector._autodownload(GITHUB_URL)
        assert isinstance(result, str)


class TestJfrogDownload:
    def test_uses_argv_list_not_shell(self, detector, tmp_path):
        # Regression: shell=True with interpolated credentials/URL was a
        # command-injection vector.
        model_path = tmp_path / "model.onnx"
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            detector._download_model_jfrog(JFROG_URL, model_path, "key", "user")
        assert run.called
        args, kwargs = run.call_args
        cmd = args[0] if args else kwargs.get("args")
        assert isinstance(cmd, list), f"expected argv list, got {type(cmd).__name__}"
        assert cmd[0] == "curl"
        assert "-u" in cmd
        assert "user:key" in cmd
        assert JFROG_URL in cmd
        assert kwargs.get("shell", False) is False

    def test_curl_failure_removes_partial_file_and_raises(self, detector, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"partial")
        assert model_path.exists()

        with patch("subprocess.run",
                   side_effect=subprocess.CalledProcessError(22, "curl")):
            with pytest.raises(RuntimeError, match="Error downloading model"):
                detector._download_model_jfrog(JFROG_URL, model_path, "k", "u")

        assert not model_path.exists()

    def test_curl_failure_with_no_partial_file_still_raises_runtime_error(
        self, detector, tmp_path
    ):
        # The cleanup branch must not explode when curl failed before writing
        # anything; only RuntimeError should escape.
        model_path = tmp_path / "model.onnx"
        assert not model_path.exists()
        with patch("subprocess.run",
                   side_effect=subprocess.CalledProcessError(22, "curl")):
            with pytest.raises(RuntimeError):
                detector._download_model_jfrog(JFROG_URL, model_path, "k", "u")


class TestSha256Verification:
    """When FILTER_YUNET_SHA256 is set, _autodownload verifies the artifact
    before returning. Hits both the cache-hit path (file already on disk) and
    the fresh-download path. Unset preserves the existing trust model.
    """

    def test_unset_env_skips_verification_on_cache_hit(
        self, detector, monkeypatch
    ):
        monkeypatch.delenv("FILTER_YUNET_SHA256", raising=False)
        monkeypatch.setattr(Path, "is_file", lambda self: True)
        with patch(
            "filter_faceblur.model.detectors.yunet_detector.verify_sha256"
        ) as mock_verify:
            detector._autodownload(GITHUB_URL)
        # Helper is invoked but with no expected hash; the helper itself
        # short-circuits on empty.
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs.get("label") == "YuNet ONNX"
        assert mock_verify.call_args.args[1] is None

    def test_set_env_invokes_verification_on_cache_hit(
        self, detector, monkeypatch
    ):
        monkeypatch.setenv("FILTER_YUNET_SHA256", "abc123")
        monkeypatch.setattr(Path, "is_file", lambda self: True)
        with patch(
            "filter_faceblur.model.detectors.yunet_detector.verify_sha256"
        ) as mock_verify:
            detector._autodownload(GITHUB_URL)
        # Expected hash threaded through to the helper.
        assert mock_verify.call_args.args[1] == "abc123"

    def test_set_env_invokes_verification_after_fresh_download(
        self, detector, monkeypatch
    ):
        monkeypatch.setenv("FILTER_YUNET_SHA256", "deadbeef")
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
        with patch.object(detector, "_download_model_opencv"), patch(
            "filter_faceblur.model.detectors.yunet_detector.verify_sha256"
        ) as mock_verify:
            detector._autodownload(GITHUB_URL)
        mock_verify.assert_called_once()
        assert mock_verify.call_args.args[1] == "deadbeef"

    def test_verification_failure_propagates(self, detector, monkeypatch):
        monkeypatch.setenv("FILTER_YUNET_SHA256", "ff" * 32)
        monkeypatch.setattr(Path, "is_file", lambda self: True)
        with patch(
            "filter_faceblur.model.detectors.yunet_detector.verify_sha256",
            side_effect=ValueError("SHA-256 mismatch"),
        ):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                detector._autodownload(GITHUB_URL)
