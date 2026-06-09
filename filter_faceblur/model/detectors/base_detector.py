# faceblur/detectors/base_detector.py
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path


def verify_sha256(path, expected, label="model"):
    """Optional SHA-256 verification for downloaded model files.

    When `expected` is falsy (None, empty string), the check is skipped —
    matching the existing un-pinned trust model. When supplied, the file is
    hashed and compared case-insensitively; on mismatch the offending file
    is removed (so the next attempt re-downloads) and a `ValueError` is
    raised. Comparison is case-insensitive so users can paste hex digests
    in either case.
    """
    if not expected:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected.lower():
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise ValueError(
            f"SHA-256 mismatch for {label} at {path}: "
            f"expected {expected}, got {actual}"
        )


class BaseDetector(ABC):
    def __init__(self, model_artifact, debug=False, api_key=None, username=None) -> None:
        """
        Initializes the BaseModel class.
        Args:
            model_artifact (str): Path to the model artifact file
            debug (bool): Debug mode flag
            api_key (str): API key for JFrog
            username (str): Username for JFrog
        """
        self.api_key = api_key
        self.username = username

    @abstractmethod
    def detect_faces(self, image):
        pass

    def get_name(self):
        return self.__class__.__name__

    def get_params(self):
        return {}
