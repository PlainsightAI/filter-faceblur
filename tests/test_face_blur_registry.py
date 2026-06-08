"""
Tests for the blurrer/detector registry and FaceBlur construction.

These tests exercise the REAL registry (no mocking of `FaceBlur` itself)
so that a valid `blurrer_name` in the config validator must also resolve
to a real class. This is the regression guard for the bug where the
validator accepted 'box' / 'median' but the registry only knew 'gaussian'.
"""

from unittest.mock import patch

import pytest

from filter_faceblur.model.blurrers.box_blur import BoxBlur
from filter_faceblur.model.blurrers.gaussian_blur import GaussianBlur
from filter_faceblur.model.blurrers.median_blur import MedianBlur
from filter_faceblur.model.detectors.yunet_detector import YuNetDetector
from filter_faceblur.model.model import FaceBlur
from filter_faceblur.model.shared import BLURRERS, DETECTORS


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


def test_face_blur_rejects_unknown_blurrer():
    with patch.object(YuNetDetector, "__init__", lambda self, *a, **kw: None):
        with pytest.raises(ValueError, match="not a valid key in the registry"):
            FaceBlur(model_artifact="unused", detector_name="yunet", blurrer_name="bogus")
