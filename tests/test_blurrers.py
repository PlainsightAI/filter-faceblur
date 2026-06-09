"""
Unit tests for the blurrer implementations.

Verifies the behaviour the registry relies on: each blurrer modifies the
face region without changing image shape/dtype, leaves the area outside
the face untouched, and degrades gracefully on empty/zero-sized regions
and zero blur_strength.
"""

import numpy as np
import pytest

from filter_faceblur.model.blurrers.base_blurrer import BaseBlurrer
from filter_faceblur.model.blurrers.box_blur import BoxBlur
from filter_faceblur.model.blurrers.gaussian_blur import GaussianBlur
from filter_faceblur.model.blurrers.median_blur import MedianBlur


ALL_BLURRERS = [BoxBlur, MedianBlur, GaussianBlur]


@pytest.fixture
def image():
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


@pytest.fixture
def face():
    return (40, 50, 80, 60)  # x, y, w, h


@pytest.mark.parametrize("blurrer_cls", ALL_BLURRERS)
class TestBlurrerContract:
    """Behavioural contract every blurrer in the registry must satisfy."""

    def test_subclasses_base(self, blurrer_cls, image, face):
        assert issubclass(blurrer_cls, BaseBlurrer)

    def test_preserves_shape_and_dtype(self, blurrer_cls, image, face):
        out = blurrer_cls().blur(image.copy(), face, blur_strength=1.0)
        assert out.shape == image.shape
        assert out.dtype == image.dtype

    def test_modifies_face_region(self, blurrer_cls, image, face):
        x, y, w, h = face
        original = image.copy()
        out = blurrer_cls().blur(image.copy(), face, blur_strength=1.0)
        # Inside the elliptical mask the pixels must change vs. random input.
        assert not np.array_equal(out[y : y + h, x : x + w], original[y : y + h, x : x + w])

    def test_leaves_outside_region_untouched(self, blurrer_cls, image, face):
        x, y, w, h = face
        original = image.copy()
        out = blurrer_cls().blur(image.copy(), face, blur_strength=1.0)
        # Top strip above the face must be byte-identical.
        assert np.array_equal(out[:y, :, :], original[:y, :, :])
        # Bottom strip below the face must be byte-identical.
        assert np.array_equal(out[y + h :, :, :], original[y + h :, :, :])

    def test_empty_region_returns_original(self, blurrer_cls, image):
        # Zero-width face -> face_region.size == 0 -> early return.
        original = image.copy()
        out = blurrer_cls().blur(image.copy(), (10, 10, 0, 50), blur_strength=1.0)
        assert np.array_equal(out, original)

    def test_blur_strength_scales_with_intensity(self, blurrer_cls, image, face):
        """Higher blur_strength should drift the face region further from the original.

        For random-noise input every blur removes high-frequency content;
        a stronger blur removes more, so its output is further (in L1) from
        the original than a weaker blur's output is.
        """
        x, y, w, h = face
        original_region = image[y : y + h, x : x + w].astype(np.int32)
        weak = blurrer_cls().blur(image.copy(), face, blur_strength=0.1)
        strong = blurrer_cls().blur(image.copy(), face, blur_strength=2.0)
        weak_drift = float(np.abs(weak[y : y + h, x : x + w].astype(np.int32) - original_region).mean())
        strong_drift = float(np.abs(strong[y : y + h, x : x + w].astype(np.int32) - original_region).mean())
        assert strong_drift > weak_drift

    def test_get_name_is_class_name(self, blurrer_cls):
        assert blurrer_cls().get_name() == blurrer_cls.__name__

    def test_get_params_returns_empty_dict(self, blurrer_cls):
        assert blurrer_cls().get_params() == {}


class TestMedianBlurKernelConstraint:
    """cv2.medianBlur requires odd ksize >= 3. The implementation must enforce this."""

    def test_small_blur_strength_does_not_crash(self, image, face):
        # blur_strength=0.01 would produce ksize=0 without the floor; medianBlur
        # would raise. Guard against that regression.
        out = MedianBlur().blur(image.copy(), face, blur_strength=0.01)
        assert out.shape == image.shape

    def test_even_scaled_kernel_is_bumped_to_odd(self, image, face):
        # 0.045 * 99 = 4.455 -> int(4) is even; the +1 bump must produce 5
        # before cv2.medianBlur is called (it rejects even ksize).
        out = MedianBlur().blur(image.copy(), face, blur_strength=0.045)
        assert out.shape == image.shape
