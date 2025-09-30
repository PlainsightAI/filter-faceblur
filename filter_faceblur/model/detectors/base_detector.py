# faceblur/detectors/base_detector.py
from abc import ABC, abstractmethod


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
