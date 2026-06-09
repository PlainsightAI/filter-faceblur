"""
Unit tests for HaarDetector and DnnDetector.

These detectors don't need YuNet's ONNX download, but they DO touch
the underlying cv2 backends (CascadeClassifier, cv2.dnn) and — for
DNN — would auto-download two large model files on first use.
Tests mock those out so the suite stays offline and fast.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from filter_faceblur.model.detectors.base_detector import BaseDetector
from filter_faceblur.model.detectors.dnn_detector import DnnDetector
from filter_faceblur.model.detectors.haar_detector import HaarDetector


@pytest.fixture
def image():
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


class TestHaarDetector:
    def test_is_base_detector(self):
        assert issubclass(HaarDetector, BaseDetector)

    def test_init_loads_bundled_cascade(self):
        # No mocks: the cascade XML ships with the opencv-python wheel,
        # so this should construct without network access.
        det = HaarDetector(model_artifact="ignored")
        assert det.cascade_path.endswith("haarcascade_frontalface_default.xml")
        assert not det.detector.empty()

    def test_detect_returns_list_of_dicts(self, image):
        det = HaarDetector(model_artifact="ignored")
        faces = det.detect_faces(image)
        assert isinstance(faces, list)
        for face in faces:
            assert set(face.keys()) >= {"bbox", "confidence"}
            assert len(face["bbox"]) == 4
            assert face["confidence"] == 1.0

    def test_detect_returns_empty_on_blank_image(self):
        det = HaarDetector(model_artifact="ignored")
        blank = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert det.detect_faces(blank) == []

    def test_raises_when_cascade_fails_to_load(self):
        with patch("cv2.CascadeClassifier") as mock_cc:
            mock_instance = MagicMock()
            mock_instance.empty.return_value = True
            mock_cc.return_value = mock_instance
            with pytest.raises(RuntimeError, match="Failed to load Haar cascade"):
                HaarDetector(model_artifact="ignored")

    def test_get_name_is_class_name(self):
        det = HaarDetector(model_artifact="ignored")
        assert det.get_name() == "HaarDetector"


class TestDnnDetector:
    def _patch_download_and_net(self):
        """Stub the two model downloads and the cv2.dnn net so __init__ runs offline."""
        return [
            patch.object(DnnDetector, "_download", return_value="/tmp/fake-model-path"),
            patch("cv2.dnn.readNetFromCaffe", return_value=MagicMock()),
        ]

    def test_is_base_detector(self):
        assert issubclass(DnnDetector, BaseDetector)

    def test_init_uses_default_urls(self):
        with patch.object(DnnDetector, "_download") as mock_dl, patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            DnnDetector(model_artifact="ignored")
            urls = [call.args[0] for call in mock_dl.call_args_list]
            assert DnnDetector.DEFAULT_PROTOTXT_URL in urls
            assert DnnDetector.DEFAULT_CAFFEMODEL_URL in urls

    def test_env_vars_override_default_urls(self, monkeypatch):
        monkeypatch.setenv("FILTER_DNN_PROTOTXT_URL", "https://example/p.prototxt")
        monkeypatch.setenv("FILTER_DNN_CAFFEMODEL_URL", "https://example/m.caffemodel")
        with patch.object(DnnDetector, "_download") as mock_dl, patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            DnnDetector(model_artifact="ignored")
            urls = [call.args[0] for call in mock_dl.call_args_list]
            assert "https://example/p.prototxt" in urls
            assert "https://example/m.caffemodel" in urls

    def test_detect_thresholds_and_returns_bbox(self, image):
        """Build a fake forward() output with two detections — one above the
        confidence floor, one below — and confirm only the strong one survives."""
        with patch.object(DnnDetector, "_download", return_value="/tmp/fake"), patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            det = DnnDetector(model_artifact="ignored")

        # Shape required by detect_faces: out[0, 0, i, [conf, x1n, y1n, x2n, y2n]]
        # indices 2..6 inclusive.
        fake_out = np.zeros((1, 1, 2, 7), dtype=np.float32)
        fake_out[0, 0, 0, 2:7] = [0.9, 0.1, 0.1, 0.5, 0.5]   # strong
        fake_out[0, 0, 1, 2:7] = [0.1, 0.0, 0.0, 0.2, 0.2]   # weak
        det.detector.forward.return_value = fake_out

        with patch("cv2.dnn.blobFromImage", return_value=np.zeros((1, 3, 300, 300))):
            faces = det.detect_faces(image, confidence_threshold=0.5)

        assert len(faces) == 1
        f = faces[0]
        assert f["confidence"] == pytest.approx(0.9)
        # Image is 200x200; bbox normalized [0.1, 0.1, 0.5, 0.5] -> [20,20,40,40] in xywh
        assert f["bbox"] == [20, 20, 80, 80]

    def test_detect_clamps_negative_coords(self, image):
        """Privacy regression: SSD output for a face at the top-left edge can
        decode to negative x1/y1. Without clamping, the emitted bbox makes the
        blurrer's image[y:y+h, x:x+w] slice empty and the face is silently
        left unblurred."""
        with patch.object(DnnDetector, "_download", return_value="/tmp/fake"), patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            det = DnnDetector(model_artifact="ignored")

        # Image is 200x200; normalized box [-0.025, -0.015, 0.225, 0.265]
        # decodes to [-5, -3, 45, 53] -> after clamp [0, 0, 45, 53] -> xywh [0,0,45,53]
        fake_out = np.zeros((1, 1, 1, 7), dtype=np.float32)
        fake_out[0, 0, 0, 2:7] = [0.95, -0.025, -0.015, 0.225, 0.265]
        det.detector.forward.return_value = fake_out

        with patch("cv2.dnn.blobFromImage", return_value=np.zeros((1, 3, 300, 300))):
            faces = det.detect_faces(image, confidence_threshold=0.5)

        assert len(faces) == 1
        x, y, w_, h_ = faces[0]["bbox"]
        assert x >= 0 and y >= 0
        assert x + w_ <= 200 and y + h_ <= 200
        assert w_ > 0 and h_ > 0

    def test_detect_clamps_overshoot_coords(self, image):
        """Symmetric case: SSD output extending past the right/bottom edge
        must be clamped to image bounds so downstream ROI consumers see
        valid coordinates."""
        with patch.object(DnnDetector, "_download", return_value="/tmp/fake"), patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            det = DnnDetector(model_artifact="ignored")

        # Box overshoots both right (x2=1.1*200=220) and bottom (y2=1.05*200=210).
        fake_out = np.zeros((1, 1, 1, 7), dtype=np.float32)
        fake_out[0, 0, 0, 2:7] = [0.95, 0.7, 0.7, 1.1, 1.05]
        det.detector.forward.return_value = fake_out

        with patch("cv2.dnn.blobFromImage", return_value=np.zeros((1, 3, 300, 300))):
            faces = det.detect_faces(image, confidence_threshold=0.5)

        assert len(faces) == 1
        x, y, w_, h_ = faces[0]["bbox"]
        assert x + w_ <= 200 and y + h_ <= 200

    def test_detect_drops_box_fully_outside_frame(self, image):
        """If the clamped box is degenerate (zero or negative area), the
        detection is dropped — there is no face to blur."""
        with patch.object(DnnDetector, "_download", return_value="/tmp/fake"), patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            det = DnnDetector(model_artifact="ignored")

        # Box entirely left of the frame.
        fake_out = np.zeros((1, 1, 1, 7), dtype=np.float32)
        fake_out[0, 0, 0, 2:7] = [0.95, -0.5, 0.1, -0.1, 0.5]
        det.detector.forward.return_value = fake_out

        with patch("cv2.dnn.blobFromImage", return_value=np.zeros((1, 3, 300, 300))):
            assert det.detect_faces(image, confidence_threshold=0.5) == []

    def test_download_wraps_network_errors(self):
        """A raw urllib URLError leaks the network stack; YuNet wraps it as a
        ValueError with the offending URL, and DNN should match that."""
        with patch("urllib.request.urlretrieve", side_effect=OSError("boom")):
            # Run only the _download method on a bare instance (skip __init__).
            det = DnnDetector.__new__(DnnDetector)
            with pytest.raises(ValueError, match="Error downloading DNN model"):
                det._download("https://example/missing.caffemodel")

    def test_detect_empty_when_all_below_threshold(self, image):
        with patch.object(DnnDetector, "_download", return_value="/tmp/fake"), patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            det = DnnDetector(model_artifact="ignored")

        fake_out = np.zeros((1, 1, 1, 7), dtype=np.float32)
        fake_out[0, 0, 0, 2:7] = [0.05, 0.1, 0.1, 0.5, 0.5]
        det.detector.forward.return_value = fake_out

        with patch("cv2.dnn.blobFromImage", return_value=np.zeros((1, 3, 300, 300))):
            assert det.detect_faces(image, confidence_threshold=0.25) == []

    def test_get_name_is_class_name(self):
        with patch.object(DnnDetector, "_download", return_value="/tmp/fake"), patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            det = DnnDetector(model_artifact="ignored")
        assert det.get_name() == "DnnDetector"


class TestDnnDetectorSha256:
    """Integration tests for SHA-256 wiring in DnnDetector.__init__.

    The hash itself is exercised by tests/test_verify_sha256.py via the
    shared helper. Here we verify that env vars flow through correctly.
    """

    def test_init_reads_sha256_env_vars(self, monkeypatch):
        monkeypatch.setenv("FILTER_DNN_PROTOTXT_SHA256", "abc")
        monkeypatch.setenv("FILTER_DNN_CAFFEMODEL_SHA256", "def")
        with patch.object(DnnDetector, "_download") as mock_dl, patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            DnnDetector(model_artifact="ignored")
            # Each _download call is _download(url, sha, label=...). Pair each
            # call's positional sha with its keyword label.
            shas_by_label = {
                call.kwargs.get("label"): call.args[1]
                for call in mock_dl.call_args_list
            }
            assert shas_by_label == {
                "DNN prototxt": "abc",
                "DNN caffemodel": "def",
            }

    def test_init_passes_none_when_env_vars_unset(self, monkeypatch):
        monkeypatch.delenv("FILTER_DNN_PROTOTXT_SHA256", raising=False)
        monkeypatch.delenv("FILTER_DNN_CAFFEMODEL_SHA256", raising=False)
        with patch.object(DnnDetector, "_download") as mock_dl, patch(
            "cv2.dnn.readNetFromCaffe", return_value=MagicMock()
        ):
            DnnDetector(model_artifact="ignored")
            for call in mock_dl.call_args_list:
                assert call.args[1] is None
