from filter_faceblur.model.detectors.yunet_detector import YuNetDetector
from filter_faceblur.model.blurrers.gaussian_blur import GaussianBlur
from filter_faceblur.model.blurrers.box_blur import BoxBlur
from filter_faceblur.model.blurrers.median_blur import MedianBlur

DETECTORS = {
    "yunet": YuNetDetector,
}

BLURRERS = {
    "gaussian": GaussianBlur,
    "box": BoxBlur,
    "median": MedianBlur,
}
