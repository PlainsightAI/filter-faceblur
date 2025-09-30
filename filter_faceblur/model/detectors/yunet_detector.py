import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os.path as osp
import urllib.request
import cv2, subprocess
from pathlib import Path

from filter_faceblur.model.detectors.base_detector import BaseDetector

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
        # Parse the model name from the URL
        model_name = model_url.split('/')[-1]      
        model_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(model_dir.parent, 'weights')    
        model_path = os.path.join(model_dir, model_name)    
        # Check if the model already exists locally
        if Path(model_path).is_file():
            print(f"Model artifact already exists at: {model_path}")    
        
            return model_path
        
        # Ensure the weights folder exists
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        if not ('jfrog' in model_url) and (api_key is None or username is None):
            self._download_model_opencv(model_url, model_path)
        
        # Parse the model name from the URL
        model_name = model_url.split('/')[-1]
        model_dir = "filter_gaf/measurements/model/weights"
        model_path = os.path.join(model_dir, model_name)
        
        # Check if the model already exists locally
        if Path(model_path).is_file():
            print(f"Model artifact already exists at: {model_path}")
            
            return model_path

        # Ensure the weights folder exists
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Try to download the model from JFrog
        print(f"Downloading model from: {model_url}")
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        curl_command = f'curl --fail -u {username}:{api_key} -L -o "{model_path}" "{model_url}"'
        
        try:
            subprocess.run(curl_command, shell=True, check=True)
            print(f"Model artifact downloaded and saved at: {model_path}")
        except subprocess.CalledProcessError as e:
            # Remove the file if it was created
            if os.path.exists(model_path):
                os.remove(model_path)    
            
            raise RuntimeError(f"Error downloading model: {model_url}")
                
        return model_path

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
            if confidence > confidence_threshold:
                box = list(map(int, detection[:4]))
                faces.append({
                    'bbox': box,
                    'confidence': float(confidence)
                })
        return faces
