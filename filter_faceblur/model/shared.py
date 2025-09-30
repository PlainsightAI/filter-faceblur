from filter_faceblur.model.detectors.yunet_detector import YuNetDetector
from filter_faceblur.model.blurrers.gaussian_blur import GaussianBlur

DETECTORS = {
    "yunet": YuNetDetector,
}

BLURRERS = {
    "gaussian": GaussianBlur,
}
