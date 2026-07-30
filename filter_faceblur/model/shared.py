from filter_faceblur.model.detectors.yunet_detector import YuNetDetector
from filter_faceblur.model.blurrers.gaussian_blur import GaussianBlur
from filter_faceblur.model.blurrers.box_blur import BoxBlur
from filter_faceblur.model.blurrers.median_blur import MedianBlur

DETECTORS = {
    "yunet": YuNetDetector,
}

# Detectors retired with the move to OpenCV 5 (openfilter 1.2.0). OpenCV 5's
# opencv-python-headless dropped `cv2.CascadeClassifier` and the bundled Haar
# cascade data (haar), and removed the `cv2.dnn` Caffe importer
# `readNetFromCaffe` (dnn). Neither has an in-core, headless-compatible
# replacement. The names are kept as soft aliases to `yunet` — the modern,
# more accurate default — so existing pipelines keep running; FaceBlur logs a
# one-time deprecation warning when a retired name is used. See
# FaceBlur._resolve_detector_alias.
DEPRECATED_DETECTORS = {
    "haar": "yunet",
    "dnn": "yunet",
}

BLURRERS = {
    "gaussian": GaussianBlur,
    "box": BoxBlur,
    "median": MedianBlur,
}
