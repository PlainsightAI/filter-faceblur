import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from filter_faceblur.model.detectors.base_detector import BaseDetector, verify_sha256


class DnnDetector(BaseDetector):
    """ResNet-SSD DNN face detector (OpenCV `cv2.dnn` Caffe backend).

    Needs two files (prototxt + caffemodel), so `model_artifact` does not fit
    the single-path contract used by YuNet. The two URLs come from
    environment variables with the standard OpenCV ResNet-SSD defaults:

      - FILTER_DNN_PROTOTXT_URL   -> deploy.prototxt
      - FILTER_DNN_CAFFEMODEL_URL -> res10_300x300_ssd_iter_140000.caffemodel

    Both are auto-downloaded into the package `weights/` directory on first
    use, mirroring `YuNetDetector._autodownload`.
    """

    DEFAULT_PROTOTXT_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/"
        "samples/dnn/face_detector/deploy.prototxt"
    )
    DEFAULT_CAFFEMODEL_URL = (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    INPUT_SIZE = (300, 300)
    MEAN_SUBTRACT = (104.0, 177.0, 123.0)

    def __init__(self, model_artifact, *args, **kwargs):
        super().__init__(model_artifact, *args, **kwargs)
        prototxt_url = os.getenv("FILTER_DNN_PROTOTXT_URL", self.DEFAULT_PROTOTXT_URL)
        caffemodel_url = os.getenv("FILTER_DNN_CAFFEMODEL_URL", self.DEFAULT_CAFFEMODEL_URL)
        # Optional SHA-256 verification; unset env vars preserve the existing
        # trust model. See base_detector.verify_sha256.
        prototxt_sha = os.getenv("FILTER_DNN_PROTOTXT_SHA256")
        caffemodel_sha = os.getenv("FILTER_DNN_CAFFEMODEL_SHA256")
        prototxt_path = self._download(prototxt_url, prototxt_sha, label="DNN prototxt")
        caffemodel_path = self._download(caffemodel_url, caffemodel_sha, label="DNN caffemodel")
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def _download(self, url, expected_sha256=None, label="DNN artifact"):
        weights_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        name = url.split("/")[-1]
        dest = weights_dir / name
        if not dest.is_file():
            print(f"Downloading DNN artifact: {url}")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                raise ValueError(f"Error downloading DNN model: {e}")
        verify_sha256(dest, expected_sha256, label=label)
        return str(dest)

    def detect_faces(self, image, confidence_threshold=0.25):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1.0, self.INPUT_SIZE, self.MEAN_SUBTRACT
        )
        self.detector.setInput(blob)
        out = self.detector.forward()
        faces = []
        for i in range(out.shape[2]):
            confidence = float(out[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue
            box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # SSD outputs are not constrained to [0, 1], so faces at the
            # frame edge can decode to negative or overshoot coords. Without
            # clamping the downstream blurrer slice image[y:y+h, x:x+w]
            # collapses to size 0 and the detection is silently dropped —
            # a privacy failure for an anonymization filter.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(
                {
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": confidence,
                }
            )
        return faces
