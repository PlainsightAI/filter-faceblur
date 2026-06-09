import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os.path as osp
import urllib.request
import cv2, subprocess
from pathlib import Path

from filter_faceblur.model.detectors.base_detector import BaseDetector, verify_sha256

class YuNetDetector(BaseDetector):
    def __init__(self, model_artifact, *args, **kwargs):
        super().__init__(model_artifact, *args, **kwargs)
        self.model_path = self._autodownload(model_url=model_artifact, api_key=self.api_key, username=self.username)
        self.detector = cv2.FaceDetectorYN_create(self.model_path, "", (0, 0))

    def _download_model_opencv(self, model_url, model_path):
        try:
            print("Downloading face detection model...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Face detection model downloaded.")
        except Exception as e:
            raise ValueError(f"Error downloading model: {e}")
            
            
    def _autodownload(self, model_url, api_key=None, username=None):
        model_name = model_url.split('/')[-1]
        weights_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / 'weights'
        model_path = weights_dir / model_name
        # Optional hardening: when FILTER_MODEL_SHA256 is set, the file (whether
        # freshly downloaded or already cached) is verified before use. Unset
        # preserves the existing trust model.
        expected_sha = os.getenv("FILTER_MODEL_SHA256")

        if model_path.is_file():
            print(f"Model artifact already exists at: {model_path}")
            verify_sha256(model_path, expected_sha, label="YuNet ONNX")
            return str(model_path)

        weights_dir.mkdir(parents=True, exist_ok=True)

        if 'jfrog' in model_url and api_key and username:
            self._download_model_jfrog(model_url, model_path, api_key, username)
        else:
            self._download_model_opencv(model_url, model_path)

        verify_sha256(model_path, expected_sha, label="YuNet ONNX")
        return str(model_path)

    def _download_model_jfrog(self, model_url, model_path, api_key, username):
        print(f"Downloading model from: {model_url}")
        # Avoid shell=True: credentials and URL are interpolated, so a stray
        # shell metacharacter (e.g. a `$` in an API key) would be executed.
        try:
            subprocess.run(
                ['curl', '--fail', '-u', f'{username}:{api_key}', '-L',
                 '-o', str(model_path), model_url],
                check=True,
            )
            print(f"Model artifact downloaded and saved at: {model_path}")
        except subprocess.CalledProcessError:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Error downloading model: {model_url}")

    def detect_faces(self, image, confidence_threshold=0.25):
        self.detector.setInputSize(image.shape[-3:-1][::-1])
        outs = self.detector.detect(image)
        return self._postprocess(image, outs, confidence_threshold)

    def _postprocess(self, image, outs, confidence_threshold=0.25):
        faces = []
        if outs[1] is None:
            return faces
        for detection in outs[1]:
            confidence = detection[-1]
            # Keep at-or-above the threshold so semantics match DnnDetector.
            # The natural reading of "minimum confidence" is "at least X".
            if confidence >= confidence_threshold:
                box = list(map(int, detection[:4]))
                faces.append({
                    'bbox': box,
                    'confidence': float(confidence)
                })
        return faces
