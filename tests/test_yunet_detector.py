"""
Unit tests for YuNetDetector._postprocess.

cv2.FaceDetectorYN's `detect()` returns `(retval, faces_array)` where each
row is `[x, y, w, h, ...landmarks..., confidence]`. The postprocess step
filters by confidence and emits the dict format the downstream code expects.
"""

import numpy as np
import pytest

from filter_faceblur.model.detectors.yunet_detector import YuNetDetector


@pytest.fixture
def detector():
    # Bypass __init__ to skip the cv2 model load — we only exercise _postprocess.
    instance = object.__new__(YuNetDetector)
    return instance


@pytest.fixture
def image():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def _detection(x, y, w, h, confidence):
    # YuNet detection row: 4 bbox + 10 landmark + 1 confidence = 15 floats.
    return np.array([x, y, w, h] + [0.0] * 10 + [confidence], dtype=np.float32)


class TestPostprocess:
    def test_returns_empty_when_outs_is_none(self, detector, image):
        assert detector._postprocess(image, (1, None)) == []

    def test_above_threshold_kept(self, detector, image):
        outs = (1, np.stack([_detection(10, 20, 30, 40, 0.9)]))
        result = detector._postprocess(image, outs, confidence_threshold=0.25)
        assert result == [{"bbox": [10, 20, 30, 40], "confidence": pytest.approx(0.9)}]

    def test_below_threshold_dropped(self, detector, image):
        outs = (1, np.stack([_detection(10, 20, 30, 40, 0.1)]))
        assert detector._postprocess(image, outs, confidence_threshold=0.25) == []

    def test_at_threshold_kept(self, detector, image):
        """Boundary case: confidence exactly equal to the threshold must be
        KEPT (>=). Aligns with DnnDetector's semantics so both detectors
        behave identically at the boundary.
        """
        outs = (1, np.stack([_detection(10, 20, 30, 40, 0.25)]))
        result = detector._postprocess(image, outs, confidence_threshold=0.25)
        assert len(result) == 1
        assert result[0]["bbox"] == [10, 20, 30, 40]
        assert result[0]["confidence"] == pytest.approx(0.25)

    def test_mixed_detections_filtered_correctly(self, detector, image):
        outs = (1, np.stack([
            _detection(0, 0, 10, 10, 0.9),    # kept
            _detection(20, 20, 10, 10, 0.24),  # below -> dropped
            _detection(40, 40, 10, 10, 0.25),  # at boundary -> kept
            _detection(60, 60, 10, 10, 0.0),   # dropped
        ]))
        result = detector._postprocess(image, outs, confidence_threshold=0.25)
        bboxes = [f["bbox"] for f in result]
        assert bboxes == [[0, 0, 10, 10], [40, 40, 10, 10]]
