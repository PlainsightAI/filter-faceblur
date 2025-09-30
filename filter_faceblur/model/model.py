import numpy as np


class FaceBlur:
    """
    A class that applies face blurring to an image using a specified face detector and blurrer.

    Args:
        detector_name (str): The name of the face detector to use. Available options are: "yunet".
        blurrer_name (str): The name of the blurrer to use. Available options are: "gaussian".

    Attributes:
        detector: An instance of the specified face detector.
        blurrer: An instance of the specified blurrer.

    Methods:
        _get_instance: Helper method to get an instance from a registry based on the provided name.
        process_frame: Applies face blurring to a frame.

    """

    def __init__(self, model_artifact: str, detector_name: str, blurrer_name: str):
        from .shared import DETECTORS, BLURRERS
        # Get the detector and blurrer classes based on the provided names
        self.detector_class = self._get_instance(DETECTORS, detector_name)
        self.blurrer_class = self._get_instance(BLURRERS, blurrer_name)
        # Initialize the detector and blurrer
        self.detector = self.detector_class(model_artifact)
        self.blurrer = self.blurrer_class()

    def _get_instance(self, registry, name: str) -> None:
        """
        Helper method to get an instance from a registry based on the provided name.

        Args:
            registry: A dictionary containing the available instances.
            name (str): The name of the instance to retrieve.

        Returns:
            An instance of the specified name.

        Raises:
            ValueError: If the provided name is not a valid key in the registry.

        """
        try:
            cls = registry[name]
        except KeyError:
            raise ValueError(
                f"'{name}' is not a valid key in the registry. Valid options are: {list(registry.keys())}"
            )
        return cls

    def process_frame(self, frame: np.ndarray, confidence_threshold: float = 0.25, blur_strength: float = 1.0) -> np.ndarray:
        """
        Applies face blurring to a frame.

        Args:
            frame: The frame to apply face blurring to.
            confidence_threshold: Minimum confidence threshold for face detection.
            blur_strength: Strength of the blur effect (0.0 = no blur, 1.0 = default blur).

        Returns:
            The frame with face blurring applied. If no face is found the original frame is returned.

        Raises:
            ValueError: If the frame is None.

        """
        # Check if image is not None
        if frame is None:
            raise ValueError("Frame is None.")
        

        faces = self.detector.detect_faces(frame, confidence_threshold)
        
        # If blur strength is 0, return original frame without blurring
        if blur_strength <= 0:
            return frame
            
        for face in faces:
            # print('face found!')
            # Extract bbox for blurring (backward compatibility)
            face_bbox = face['bbox'] if isinstance(face, dict) else face
            output = self.blurrer.blur(frame, face_bbox, blur_strength)

        if faces:
            return output
        else:
            return frame
