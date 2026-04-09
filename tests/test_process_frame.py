"""
Tests for FaceBlur.process_frame() and the filter's single-detection guarantee.

These tests verify that:
- process_frame() uses the faces list it receives (no internal detection)
- The filter pipeline calls detect_faces exactly once per frame
- The faces returned by detect_faces are passed through to process_frame
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from openfilter.filter_runtime.filter import Frame
from filter_faceblur.filter import FilterFaceblur, FilterFaceblurConfig
from filter_faceblur.model.model import FaceBlur


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    """A 200x200 BGR image."""
    return np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame(sample_image):
    """A Frame wrapping the sample image."""
    return Frame(sample_image, {"meta": {"id": 1}}, "BGR")


@pytest.fixture
def mock_blurrer():
    blurrer = MagicMock()
    # blur returns the image it receives (in-place modification pattern)
    blurrer.blur.side_effect = lambda image, bbox, strength: image
    return blurrer


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.detect_faces.return_value = []
    return detector


@pytest.fixture
def face_blur(mock_detector, mock_blurrer):
    """A FaceBlur instance with mocked detector and blurrer."""
    with patch.object(FaceBlur, "__init__", lambda self, *a, **kw: None):
        fb = FaceBlur.__new__(FaceBlur)
        fb.detector = mock_detector
        fb.blurrer = mock_blurrer
        return fb


@pytest.fixture
def dict_faces():
    """Face detections in dict format (as returned by YuNet)."""
    return [
        {"bbox": [10, 20, 50, 60], "confidence": 0.95},
        {"bbox": [100, 100, 40, 40], "confidence": 0.80},
    ]


@pytest.fixture
def list_faces():
    """Face detections in plain list format (backward compat)."""
    return [
        [10, 20, 50, 60],
        [100, 100, 40, 40],
    ]


# ---------------------------------------------------------------------------
# FaceBlur.process_frame — unit tests
# ---------------------------------------------------------------------------

class TestProcessFrame:
    """Tests for FaceBlur.process_frame() with the new required faces parameter."""

    def test_does_not_call_detect_faces(self, face_blur, sample_image, dict_faces):
        """process_frame must NOT call detect_faces internally."""
        face_blur.process_frame(sample_image, dict_faces, blur_strength=1.0)
        face_blur.detector.detect_faces.assert_not_called()

    def test_blurs_each_detected_face_dict_format(self, face_blur, sample_image, dict_faces):
        """Each face in the list should be passed to the blurrer (dict format)."""
        face_blur.process_frame(sample_image, dict_faces, blur_strength=1.0)
        assert face_blur.blurrer.blur.call_count == len(dict_faces)
        for face_dict in dict_faces:
            face_blur.blurrer.blur.assert_any_call(
                sample_image, face_dict["bbox"], 1.0
            )

    def test_blurs_each_detected_face_list_format(self, face_blur, sample_image, list_faces):
        """Each face in the list should be passed to the blurrer (list format)."""
        face_blur.process_frame(sample_image, list_faces, blur_strength=1.0)
        assert face_blur.blurrer.blur.call_count == len(list_faces)
        for face_bbox in list_faces:
            face_blur.blurrer.blur.assert_any_call(
                sample_image, face_bbox, 1.0
            )

    def test_empty_faces_returns_original(self, face_blur, sample_image):
        """When faces is empty, return the original frame unchanged."""
        result = face_blur.process_frame(sample_image, [], blur_strength=1.0)
        assert result is sample_image
        face_blur.blurrer.blur.assert_not_called()

    def test_none_frame_raises(self, face_blur, dict_faces):
        """A None frame must raise ValueError."""
        with pytest.raises(ValueError, match="Frame is None"):
            face_blur.process_frame(None, dict_faces)

    def test_zero_blur_strength_skips_blurring(self, face_blur, sample_image, dict_faces):
        """blur_strength <= 0 should skip blurring entirely."""
        result = face_blur.process_frame(sample_image, dict_faces, blur_strength=0.0)
        assert result is sample_image
        face_blur.blurrer.blur.assert_not_called()

    def test_negative_blur_strength_skips_blurring(self, face_blur, sample_image, dict_faces):
        """Negative blur_strength should also skip blurring."""
        result = face_blur.process_frame(sample_image, dict_faces, blur_strength=-1.0)
        assert result is sample_image
        face_blur.blurrer.blur.assert_not_called()

    def test_single_face(self, face_blur, sample_image):
        """Single face detection works correctly."""
        faces = [{"bbox": [10, 10, 30, 30], "confidence": 0.99}]
        result = face_blur.process_frame(sample_image, faces, blur_strength=0.5)
        face_blur.blurrer.blur.assert_called_once_with(sample_image, [10, 10, 30, 30], 0.5)

    def test_blur_strength_passed_through(self, face_blur, sample_image):
        """Custom blur strength is forwarded to the blurrer."""
        faces = [{"bbox": [0, 0, 10, 10], "confidence": 0.9}]
        face_blur.process_frame(sample_image, faces, blur_strength=2.5)
        face_blur.blurrer.blur.assert_called_once_with(sample_image, [0, 0, 10, 10], 2.5)


# ---------------------------------------------------------------------------
# FilterFaceblur.process — single-detection guarantee
# ---------------------------------------------------------------------------

class TestFilterSingleDetection:
    """Verify the filter calls detect_faces exactly once per image frame."""

    def _setup_filter(self, mock_face_blur, **config_overrides):
        """Helper to create a configured filter with a mocked FaceBlur."""
        config_data = {
            "detector_name": "yunet",
            "blurrer_name": "gaussian",
            "blur_strength": 1.0,
            "detection_confidence_threshold": 0.5,
            "debug": False,
            "blur_enabled": True,
            "forward_upstream_data": True,
            "include_face_coordinates": True,
        }
        config_data.update(config_overrides)
        config = FilterFaceblur.normalize_config(config_data)
        filt = FilterFaceblur(config=config)
        with patch("filter_faceblur.model.FaceBlur") as cls:
            cls.return_value = mock_face_blur
            filt.setup(config)
        return filt

    def test_detect_faces_called_once_per_frame(self, sample_frame):
        """detect_faces must be called exactly once per image frame."""
        mock_fb = MagicMock()
        detected = [{"bbox": [10, 20, 50, 60], "confidence": 0.9}]
        mock_fb.detector.detect_faces.return_value = detected
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        filt = self._setup_filter(mock_fb)
        filt.process({"main": sample_frame})

        mock_fb.detector.detect_faces.assert_called_once()

    def test_detected_faces_passed_to_process_frame(self, sample_frame):
        """The exact faces list from detect_faces must be forwarded to process_frame."""
        mock_fb = MagicMock()
        detected = [
            {"bbox": [10, 20, 50, 60], "confidence": 0.9},
            {"bbox": [80, 80, 30, 30], "confidence": 0.7},
        ]
        mock_fb.detector.detect_faces.return_value = detected
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        filt = self._setup_filter(mock_fb)
        filt.process({"main": sample_frame})

        # Second arg to process_frame should be the detected faces list
        args = mock_fb.process_frame.call_args
        assert args[0][1] is detected

    def test_multiple_topics_each_detected_once(self, sample_frame):
        """Each image topic should trigger exactly one detect_faces call."""
        mock_fb = MagicMock()
        mock_fb.detector.detect_faces.return_value = []
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        filt = self._setup_filter(mock_fb)
        filt.process({"main": sample_frame, "cam2": sample_frame, "cam3": sample_frame})

        assert mock_fb.detector.detect_faces.call_count == 3
        assert mock_fb.process_frame.call_count == 3

    def test_blur_disabled_still_detects_once(self, sample_frame):
        """With blur disabled, detection still happens once but process_frame is not called."""
        mock_fb = MagicMock()
        detected = [{"bbox": [10, 20, 50, 60], "confidence": 0.9}]
        mock_fb.detector.detect_faces.return_value = detected

        filt = self._setup_filter(mock_fb, blur_enabled=False)
        filt.process({"main": sample_frame})

        mock_fb.detector.detect_faces.assert_called_once()
        mock_fb.process_frame.assert_not_called()

    def test_zero_strength_still_detects_once(self, sample_frame):
        """With blur_strength=0, detection still happens once but process_frame is not called."""
        mock_fb = MagicMock()
        detected = [{"bbox": [10, 20, 50, 60], "confidence": 0.9}]
        mock_fb.detector.detect_faces.return_value = detected

        filt = self._setup_filter(mock_fb, blur_strength=0.0)
        filt.process({"main": sample_frame})

        mock_fb.detector.detect_faces.assert_called_once()
        mock_fb.process_frame.assert_not_called()

    def test_face_metadata_matches_single_detection(self, sample_frame):
        """Face metadata in output should come from the single detect_faces call."""
        mock_fb = MagicMock()
        detected = [
            {"bbox": [10, 20, 50, 60], "confidence": 0.95},
            {"bbox": [80, 80, 30, 30], "confidence": 0.70},
        ]
        mock_fb.detector.detect_faces.return_value = detected
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        filt = self._setup_filter(mock_fb, include_face_coordinates=True)
        output = filt.process({"main": sample_frame})

        frame_data = output["main"].data
        assert frame_data["faces_detected"] == 2
        assert frame_data["face_coordinates"] is detected
        assert len(frame_data["face_details"]) == 2
        assert frame_data["face_details"][0]["confidence"] == 0.95
        assert frame_data["face_details"][1]["confidence"] == 0.70

    def test_no_faces_detected_metadata(self, sample_frame):
        """When no faces are detected, metadata reflects zero detections."""
        mock_fb = MagicMock()
        mock_fb.detector.detect_faces.return_value = []
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        filt = self._setup_filter(mock_fb, include_face_coordinates=True)
        output = filt.process({"main": sample_frame})

        frame_data = output["main"].data
        assert frame_data["faces_detected"] == 0
        assert frame_data["face_coordinates"] == []
        assert "face_details" not in frame_data

    def test_non_image_frames_skip_detection(self, sample_frame):
        """Non-image frames should not trigger detection at all."""
        mock_fb = MagicMock()
        mock_fb.detector.detect_faces.return_value = []
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        data_only = Frame({"some": "data"})

        filt = self._setup_filter(mock_fb, forward_upstream_data=True)
        output = filt.process({"main": sample_frame, "data": data_only})

        # Only called for the image frame
        mock_fb.detector.detect_faces.assert_called_once()
        # Data-only frame is forwarded
        assert output["data"] is data_only

    def test_none_frame_skips_detection(self):
        """None frames should not trigger detection."""
        mock_fb = MagicMock()
        mock_fb.detector.detect_faces.return_value = []

        filt = self._setup_filter(mock_fb, forward_upstream_data=True)
        output = filt.process({"main": None})

        mock_fb.detector.detect_faces.assert_not_called()
        assert output["main"] is None

    def test_crop_filter_detections_format(self, sample_frame):
        """Detections in meta should use [x1, y1, x2, y2] format for crop filter compatibility."""
        mock_fb = MagicMock()
        detected = [{"bbox": [10, 20, 50, 60], "confidence": 0.9}]
        mock_fb.detector.detect_faces.return_value = detected
        mock_fb.process_frame.return_value = sample_frame.rw_bgr.image

        filt = self._setup_filter(mock_fb, include_face_coordinates=True)
        output = filt.process({"main": sample_frame})

        detections = output["main"].data["meta"]["detections"]
        assert len(detections) == 1
        assert detections[0]["class"] == "face"
        # [x, y, w, h] -> [x1, y1, x2, y2] = [10, 20, 60, 80]
        assert detections[0]["rois"] == [10, 20, 60, 80]
        assert detections[0]["confidence"] == 0.9
