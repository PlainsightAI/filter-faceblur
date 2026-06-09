import cv2

from filter_faceblur.model.detectors.base_detector import BaseDetector


class HaarDetector(BaseDetector):
    """Haar-cascade face detector backed by OpenCV's bundled frontal-face XML.

    `model_artifact` is accepted for interface parity with the other detectors
    but is ignored: the cascade ships with the `opencv-python` wheel, so no
    download is required. Override `DEFAULT_CASCADE` (or the resolved
    `self.cascade_path` after construction) to point at a different XML.
    """

    DEFAULT_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self, model_artifact, *args, **kwargs):
        super().__init__(model_artifact, *args, **kwargs)
        self.cascade_path = self.DEFAULT_CASCADE
        self.detector = cv2.CascadeClassifier(self.cascade_path)
        if self.detector.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {self.cascade_path}"
            )

    def detect_faces(self, image, confidence_threshold=0.25):
        # Haar cascades are non-probabilistic; confidence_threshold is part of
        # the contract but has no native meaning here. Every detection is
        # reported with confidence=1.0 so downstream code treats them uniformly.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )
        return [
            {"bbox": [int(x), int(y), int(w), int(h)], "confidence": 1.0}
            for (x, y, w, h) in rects
        ]
