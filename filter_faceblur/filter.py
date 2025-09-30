import logging, os
from typing import Dict

logger = logging.getLogger(__name__)

from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# Read the environment variables and provide default values if they are not present
MODEL_ARTIFACTORY_URL =  os.getenv(
    "MODEL_ARTIFACTORY_URL", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
)
MODEL_ARTIFACT_NAME =  os.getenv(
    "MODEL_ARTIFACT_NAME", "face_detection_yunet_2023mar.onnx"
)
DETECTOR_NAME = os.getenv(
    "DETECTOR_NAME", "yunet"
)
BLURRER_NAME = os.getenv(
    "BLURRER_NAME", "gaussian"
)

__all__ = ['FilterFaceblurConfig', 'FilterFaceblur']

logger = logging.getLogger(__name__)


class FilterFaceblurConfig(FilterConfig):
    """Configuration for the Face Blur filter."""
    detector_name:                     str = "yunet"
    blurrer_name:                      str = "gaussian"
    model_artifactory_url:             str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    model_artifact_name:               str = "face_detection_yunet_2023mar.onnx"
    blur_strength:                     float = 1.0
    blur_enabled:                      bool = True  # Enable/disable face blurring
    detection_confidence_threshold:    float = 0.25  # Minimum confidence for face detection
    debug:                             bool = False
    forward_upstream_data:             bool = True  # Forward data from upstream filters
    include_face_coordinates:          bool = True  # Include face detection coordinates in frame data


class FilterFaceblur(Filter):
    """
    A filter that detects and blurs faces in video frames using OpenCV's YuNet face detector.
    
    Features:
    - Face detection using YuNet model
    - Gaussian blur for face anonymization (configurable strength)
    - Option to disable blurring while still detecting faces
    - Configurable blur strength and confidence threshold
    - Support for different detector and blurrer types
    - Forward data from upstream filters (configurable)
    - Face coordinates and detection metadata in frame data (configurable)
    - Multi-topic processing (processes all image topics)
    """

    @classmethod
    def normalize_config(cls, config: FilterFaceblurConfig):
        """
        Convert environment variables to correct data types, apply checks, etc.
        """
        # Call parent class normalize_config first to handle sources/outputs parsing
        config = super().normalize_config(config)
        
        # Handle nested config structure
        if isinstance(config, dict) and 'config' in config:
            config = config['config']
            
        # Load environment variables if config is empty or missing values
        if isinstance(config, dict):
            # Load from environment variables if not provided in config
            env_mapping = {
                'detector_name': str,
                'blurrer_name': str,
                'model_artifactory_url': str,
                'model_artifact_name': str,
                'blur_strength': float,
                'blur_enabled': str,
                'detection_confidence_threshold': float,
                'debug': str,
                'forward_upstream_data': str,
                'include_face_coordinates': str,
            }
            
            for key, expected_type in env_mapping.items():
                env_key = f"FILTER_{key.upper()}"
                if key not in config and env_key in os.environ:
                    env_val = os.environ[env_key]
                    if expected_type is float:
                        config[key] = float(env_val)
                    elif expected_type is str and key in ['debug', 'forward_upstream_data', 'include_face_coordinates', 'blur_enabled']:
                        if env_val.lower() not in ['true', 'false']:
                            raise ValueError(f"Invalid {key} mode: {env_val}. It should be True or False.")
                        config[key] = env_val.lower() == 'true'
                    else:
                        config[key] = env_val
            
            # Convert numeric strings to proper types
            if 'blur_strength' in config and isinstance(config['blur_strength'], str):
                config['blur_strength'] = float(config['blur_strength'])
            if 'detection_confidence_threshold' in config and isinstance(config['detection_confidence_threshold'], str):
                config['detection_confidence_threshold'] = float(config['detection_confidence_threshold'])
            if 'debug' in config and isinstance(config['debug'], str):
                if config['debug'].lower() not in ['true', 'false']:
                    raise ValueError(f"Invalid debug mode: {config['debug']}. It should be True or False.")
                config['debug'] = config['debug'].lower() == 'true'
            if 'forward_upstream_data' in config and isinstance(config['forward_upstream_data'], str):
                config['forward_upstream_data'] = config['forward_upstream_data'].lower() == 'true'
            if 'include_face_coordinates' in config and isinstance(config['include_face_coordinates'], str):
                config['include_face_coordinates'] = config['include_face_coordinates'].lower() == 'true'
            if 'blur_enabled' in config and isinstance(config['blur_enabled'], str):
                if config['blur_enabled'].lower() not in ['true', 'false']:
                    raise ValueError(f"Invalid blur_enabled mode: {config['blur_enabled']}. It should be True or False.")
                config['blur_enabled'] = config['blur_enabled'].lower() == 'true'

        # Convert to FilterFaceblurConfig
        config = FilterFaceblurConfig(**config)
        
        # Validate detector name
        valid_detectors = ['yunet', 'haar', 'dnn']
        if config.detector_name not in valid_detectors:
            raise ValueError(f"Invalid detector_name: {config.detector_name}. Must be one of {valid_detectors}")
        
        # Validate blurrer name
        valid_blurrers = ['gaussian', 'box', 'median']
        if config.blurrer_name not in valid_blurrers:
            raise ValueError(f"Invalid blurrer_name: {config.blurrer_name}. Must be one of {valid_blurrers}")
        
        # Validate blur strength (allow 0 for disabling blur)
        if config.blur_strength < 0:
            raise ValueError("Blur strength must be non-negative")
        
        # Validate detection confidence threshold
        if not 0 <= config.detection_confidence_threshold <= 1:
            raise ValueError("Detection confidence threshold must be between 0 and 1")
        
        # Validate debug mode
        if not isinstance(config.debug, bool):
            raise ValueError(f"Invalid debug mode: {config.debug}. It should be True or False.")
        
        # Validate forward_upstream_data mode
        if not isinstance(config.forward_upstream_data, bool):
            raise ValueError(f"Invalid forward_upstream_data mode: {config.forward_upstream_data}. It should be True or False.")
        
        # Validate include_face_coordinates mode
        if not isinstance(config.include_face_coordinates, bool):
            raise ValueError(f"Invalid include_face_coordinates mode: {config.include_face_coordinates}. It should be True or False.")
        
        # Validate blur_enabled mode
        if not isinstance(config.blur_enabled, bool):
            raise ValueError(f"Invalid blur_enabled mode: {config.blur_enabled}. It should be True or False.")
        
        return config

    def setup(self, config):
        from filter_faceblur.model import FaceBlur
        
        # Store config
        self.config = config
        
        # Use config parameters or fall back to environment variables
        model_artifactory_url = getattr(config, 'model_artifactory_url', MODEL_ARTIFACTORY_URL)
        model_artifact_name = getattr(config, 'model_artifact_name', MODEL_ARTIFACT_NAME)
        detector_name = getattr(config, 'detector_name', DETECTOR_NAME)
        blurrer_name = getattr(config, 'blurrer_name', BLURRER_NAME)
        
        # Build model artifact path
        model_artifact = os.path.join(model_artifactory_url, model_artifact_name)
        
        # Initialize face blur with configuration
        self.face_blur = FaceBlur(
            model_artifact=model_artifact, 
            detector_name=detector_name, 
            blurrer_name=blurrer_name
        )
        
        # Note: isinstance check removed as it causes issues with mocking in tests
        
        # Set additional configuration if available
        if hasattr(config, 'blur_strength'):
            self.blur_strength = config.blur_strength
        else:
            self.blur_strength = 1.0
            
        if hasattr(config, 'detection_confidence_threshold'):
            self.detection_confidence_threshold = config.detection_confidence_threshold
        else:
            self.detection_confidence_threshold = 0.25
            
            
        if hasattr(config, 'debug'):
            self.debug = config.debug
        else:
            self.debug = False
        
        logger.info(f"FilterFaceblur setup completed. Detector: {detector_name}, Blurrer: {blurrer_name}, Debug: {self.debug}")

    def shutdown(self):
        pass  # TODO: shutdown

    def process(self, frames: Dict) -> Dict:
        # Initialize output frames dictionary
        output_frames = {}
        
        # Process all frames that contain images
        for topic, frame in frames.items():
            if frame is None or not frame.has_image:
                # Forward non-image frames as-is if upstream forwarding is enabled
                if self.config.forward_upstream_data:
                    output_frames[topic] = frame
                continue
                
            # Detect faces and get coordinates
            faces = self.face_blur.detector.detect_faces(frame.rw_bgr.image, self.config.detection_confidence_threshold)
            
            if self.config.debug:
                logger.info(f"Detected {len(faces)} faces in {topic} frame")
            
            # Apply face blurring (only if enabled and strength > 0)
            if self.config.blur_enabled and self.config.blur_strength > 0:
                if self.config.debug:
                    logger.info(f"Applying blur with strength {self.config.blur_strength} to {topic} frame")
                output = self.face_blur.process_frame(frame.rw_bgr.image, self.config.detection_confidence_threshold, self.config.blur_strength)
            else:
                # Skip blurring but still return the original frame
                if self.config.debug:
                    blur_reason = "disabled" if not self.config.blur_enabled else f"strength={self.config.blur_strength}"
                    logger.info(f"Blurring skipped ({blur_reason}) for {topic} frame")
                output = frame.rw_bgr.image
            
            # Prepare frame data
            frame_data = frame.data.copy() if frame.data else {}
            
            # Include face detection information if enabled
            if self.config.include_face_coordinates:
                frame_data['faces_detected'] = len(faces)
                frame_data['face_coordinates'] = faces
                
                # Add individual face information if faces were detected
                if faces:
                    frame_data['face_details'] = []
                    # Initialize meta dict for crop filter compatibility
                    if 'meta' not in frame_data:
                        frame_data['meta'] = {}
                    
                    # Prepare detections in format expected by FilterCrop
                    detections = []
                    for i, face_data in enumerate(faces):
                        # Handle both old format (list) and new format (dict)
                        if isinstance(face_data, dict):
                            face_box = face_data['bbox']
                            confidence = face_data['confidence']
                        else:
                            # Backward compatibility for old format
                            face_box = face_data
                            confidence = None
                        
                        x, y, w, h = face_box
                        
                        # Convert to [x1, y1, x2, y2] format for crop filter
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        
                        # Create detection object for crop filter
                        detection = {
                            'class': 'face',  # detection_class_field
                            'rois': [x1, y1, x2, y2]  # detection_roi_field
                        }
                        
                        # Add confidence if available
                        if confidence is not None:
                            detection['confidence'] = confidence
                        
                        detections.append(detection)
                        
                        # Also create detailed face info for backward compatibility
                        face_info = {
                            'face_id': i,
                            'bounding_box': {
                                'x': int(x),
                                'y': int(y), 
                                'width': int(w),
                                'height': int(h)
                            },
                            'center': {
                                'x': int(x + w/2),
                                'y': int(y + h/2)
                            }
                        }
                        
                        # Add confidence if available
                        if confidence is not None:
                            face_info['confidence'] = confidence
                        
                        frame_data['face_details'].append(face_info)
                    
                    # Store detections in meta for crop filter
                    frame_data['meta']['detections'] = detections
            
            # Create processed frame
            processed_frame = Frame(image=output, data=frame_data, format='BGR')
            output_frames[topic] = processed_frame
        
        # Ensure main topic comes first in the output dictionary
        if 'main' in output_frames:
            main_frame = output_frames.pop('main')
            return {'main': main_frame, **output_frames}
        
        return output_frames

    def get_name(self):
        return "FaceBlurFilter"

    def get_description(self):
        return "This filter blurs faces in the frame."


if __name__ == '__main__':
    FilterFaceblur.run()
