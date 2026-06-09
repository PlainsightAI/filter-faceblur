import numpy as np


class FaceBlur:
    """
    A class that applies face blurring to an image using a specified face detector and blurrer.

    Args:
        detector_name (str): The name of the face detector to use. Available options are: "yunet", "haar", "dnn".
        blurrer_name (str): The name of the blurrer to use. Available options are: "gaussian", "box", "median".

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

    @staticmethod
    def clamp_faces_to_frame(faces: list, image_shape) -> list:
        """Clamp face bboxes to image bounds and drop degenerate detections.

        Detectors can emit bboxes that extend past the frame edges (negative
        coords for top-left faces, overshoot for bottom-right faces). The
        downstream blurrer slices `image[y:y+h, x:x+w]`, which collapses to
        size 0 on a negative start — silently leaving the detected face
        unblurred. For an anonymization filter that is a privacy failure.

        Accepts the dict format (`{'bbox': [x, y, w, h], 'confidence': ...}`)
        or the bare 4-tuple format and preserves whichever was passed in. A
        face whose clamped box has zero or negative area (fully outside the
        frame) is dropped.
        """
        if image_shape is None or len(image_shape) < 2:
            return list(faces)
        h, w = int(image_shape[0]), int(image_shape[1])
        clamped = []
        for face in faces:
            if isinstance(face, dict):
                bbox = face.get('bbox')
            else:
                bbox = face
            if bbox is None or len(bbox) < 4:
                continue
            x, y, fw, fh = bbox[0], bbox[1], bbox[2], bbox[3]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x) + int(fw))
            y2 = min(h, int(y) + int(fh))
            if x2 <= x1 or y2 <= y1:
                continue
            new_bbox = [x1, y1, x2 - x1, y2 - y1]
            if isinstance(face, dict):
                entry = dict(face)
                entry['bbox'] = new_bbox
                clamped.append(entry)
            else:
                clamped.append(new_bbox)
        return clamped

    def process_frame(self, frame: np.ndarray, faces: list, blur_strength: float = 1.0) -> np.ndarray:
        """
        Applies face blurring to a frame.

        Args:
            frame: The frame to apply face blurring to.
            faces: List of detected faces to blur.
            blur_strength: Strength of the blur effect (0.0 = no blur, 1.0 = default blur).

        Returns:
            The frame with face blurring applied. If no face is found the original frame is returned.

        Raises:
            ValueError: If the frame is None.

        """
        # Check if image is not None
        if frame is None:
            raise ValueError("Frame is None.")

        # If blur strength is 0, return original frame without blurring
        if blur_strength <= 0:
            return frame

        # Defense-in-depth: clamp regardless of caller. filter.py also clamps
        # before metadata building so downstream ROI consumers see the same
        # values; this guard keeps direct callers (e.g. scripts/model_usage.py)
        # safe.
        faces = self.clamp_faces_to_frame(faces, frame.shape)

        # Thread the accumulated `output` through each blurrer call. Every
        # current blurrer mutates in place, so passing `frame` would have
        # worked, but a future blurrer that returns a copy would silently
        # drop every face except the last.
        output = frame
        for face in faces:
            face_bbox = face['bbox'] if isinstance(face, dict) else face
            output = self.blurrer.blur(output, face_bbox, blur_strength)
        return output
