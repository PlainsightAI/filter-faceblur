"""
Tests for the blurrer/detector registry and FaceBlur construction.

These tests exercise the REAL registry (no mocking of `FaceBlur` itself)
so that a valid `blurrer_name` in the config validator must also resolve
to a real class. This is the regression guard for the bug where the
validator accepted 'box' / 'median' but the registry only knew 'gaussian'.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from filter_faceblur.model.blurrers.box_blur import BoxBlur
from filter_faceblur.model.blurrers.gaussian_blur import GaussianBlur
from filter_faceblur.model.blurrers.median_blur import MedianBlur
from filter_faceblur.model.detectors.dnn_detector import DnnDetector
from filter_faceblur.model.detectors.haar_detector import HaarDetector
from filter_faceblur.model.detectors.yunet_detector import YuNetDetector
from filter_faceblur.model.model import FaceBlur
from filter_faceblur.model.shared import BLURRERS, DETECTORS


# Map each detector class to a patch that skips its __init__ (model download,
# cv2 setup) so the registry-level tests don't need network or model files.
DETECTOR_INIT_PATCH_TARGETS = [YuNetDetector, HaarDetector, DnnDetector]


@contextmanager
def all_detectors_stubbed():
    """Stub every detector's __init__ so FaceBlur(...) only exercises the
    registry lookup, not the heavy model setup."""
    patches = [
        patch.object(cls, "__init__", lambda self, *a, **kw: None)
        for cls in DETECTOR_INIT_PATCH_TARGETS
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


class TestBlurrerRegistry:
    def test_gaussian_resolves(self):
        assert BLURRERS["gaussian"] is GaussianBlur

    def test_box_resolves(self):
        assert BLURRERS["box"] is BoxBlur

    def test_median_resolves(self):
        assert BLURRERS["median"] is MedianBlur

    def test_no_unexpected_entries(self):
        assert set(BLURRERS) == {"gaussian", "box", "median"}


class TestDetectorRegistry:
    def test_yunet_resolves(self):
        assert DETECTORS["yunet"] is YuNetDetector

    def test_haar_resolves(self):
        assert DETECTORS["haar"] is HaarDetector

    def test_dnn_resolves(self):
        assert DETECTORS["dnn"] is DnnDetector

    def test_no_unexpected_entries(self):
        assert set(DETECTORS) == {"yunet", "haar", "dnn"}


@pytest.mark.parametrize("blurrer_name", ["gaussian", "box", "median"])
def test_face_blur_constructs_with_each_blurrer(blurrer_name):
    """Regression: every blurrer_name accepted by the validator must
    instantiate against the real BLURRERS registry.

    Patches only YuNetDetector to skip the model download — the blurrer
    side goes through the real registry lookup we care about.
    """
    with patch.object(YuNetDetector, "__init__", lambda self, *a, **kw: None):
        face_blur = FaceBlur(
            model_artifact="unused",
            detector_name="yunet",
            blurrer_name=blurrer_name,
        )
    assert isinstance(face_blur.blurrer, BLURRERS[blurrer_name])


@pytest.mark.parametrize("detector_name", ["yunet", "haar", "dnn"])
def test_face_blur_constructs_with_each_detector(detector_name):
    """Regression: every detector_name accepted by the validator must
    instantiate against the real DETECTORS registry.

    Same class of bug as the blurrer side — validator advertised
    ['yunet', 'haar', 'dnn'] but only yunet was registered, so haar/dnn
    crashed at construction. This guards both directions.
    """
    with all_detectors_stubbed():
        face_blur = FaceBlur(
            model_artifact="unused",
            detector_name=detector_name,
            blurrer_name="gaussian",
        )
    assert isinstance(face_blur.detector, DETECTORS[detector_name])


def test_face_blur_rejects_unknown_blurrer():
    with patch.object(YuNetDetector, "__init__", lambda self, *a, **kw: None):
        with pytest.raises(ValueError, match="not a valid key in the registry"):
            FaceBlur(model_artifact="unused", detector_name="yunet", blurrer_name="bogus")


def test_face_blur_rejects_unknown_detector():
    with all_detectors_stubbed():
        with pytest.raises(ValueError, match="not a valid key in the registry"):
            FaceBlur(model_artifact="unused", detector_name="bogus", blurrer_name="gaussian")
