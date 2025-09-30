#!/usr/bin/env python3
"""
Filter Face Blur Usage Example

This script demonstrates how to use FilterFaceblur in a pipeline:
VideoIn → FilterFaceblur → VideoOut + Webvis

Prerequisites:
- Sample video file (specify via VIDEO_INPUT environment variable)
- GPU recommended for optimal performance

Environment Variables:
- VIDEO_INPUT: Input video file path (default: ./data/video-01.mp4)
- OUTPUT_VIDEO_PATH: Output video file path (default: ./output/output.mp4)
- OUTPUT_FPS: Output video frames per second (default: 30)
- WEBVIS_PORT: Port for Webvis visualization (default: 8000)
- FILTER_DETECTOR_NAME: Face detector type (default: yunet)
- FILTER_BLURRER_NAME: Blur algorithm type (default: gaussian)
- FILTER_MODEL_ARTIFACTORY_URL: Model repository URL (default: OpenCV Zoo)
- FILTER_MODEL_ARTIFACT_NAME: Model filename (default: face_detection_yunet_2023mar.onnx)
- FILTER_BLUR_STRENGTH: Blur intensity (default: 1.0)
- FILTER_BLUR_ENABLED: Enable/disable face blurring (default: True)
- FILTER_DETECTION_CONFIDENCE_THRESHOLD: Minimum confidence for face detection (default: 0.25)
- FILTER_DEBUG: Enable debug logging (default: False)
- FILTER_FORWARD_UPSTREAM_DATA: Forward data from upstream filters (default: True)
- FILTER_INCLUDE_FACE_COORDINATES: Include face coordinates in frame data (default: True)
"""

import logging
import os
import sys

# Add the filter module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import OpenFilter components
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.video_out import VideoOut
from openfilter.filter_runtime.filters.webvis import Webvis
from openfilter.filter_runtime.filters.util import Util

# Import our face blur filter
from filter_faceblur import FilterFaceblur, FilterFaceblurConfig


def load_config():
    """
    Load configuration from environment variables for the Face Blur filter.
    """
    config = {
        "detector_name": os.getenv("FILTER_DETECTOR_NAME", "yunet"),
        "blurrer_name": os.getenv("FILTER_BLURRER_NAME", "gaussian"),
        "model_artifactory_url": os.getenv("FILTER_MODEL_ARTIFACTORY_URL", 
                                          "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"),
        "model_artifact_name": os.getenv("FILTER_MODEL_ARTIFACT_NAME", "face_detection_yunet_2023mar.onnx"),
        "blur_strength": float(os.getenv("FILTER_BLUR_STRENGTH", "1.0")),
        "blur_enabled": os.getenv("FILTER_BLUR_ENABLED", "True").lower() == "true",
        "detection_confidence_threshold": float(os.getenv("FILTER_DETECTION_CONFIDENCE_THRESHOLD", "0.25")),
        "debug": os.getenv("FILTER_DEBUG", "False").lower() == "true",
        "forward_upstream_data": os.getenv("FILTER_FORWARD_UPSTREAM_DATA", "True").lower() == "true",
        "include_face_coordinates": os.getenv("FILTER_INCLUDE_FACE_COORDINATES", "True").lower() == "true",
    }
    
    return config


def main():
    """Run the FilterFaceblur pipeline."""
    
    # Configuration from environment variables
    VIDEO_INPUT = os.getenv("VIDEO_INPUT", "./data/video-01.mp4")
    FPS = int(os.getenv("OUTPUT_FPS", "30"))
    WEBVIS_PORT = int(os.getenv("WEBVIS_PORT", "8000"))
    
    # Parse input video name for output filename
    input_basename = os.path.splitext(os.path.basename(VIDEO_INPUT))[0]
    default_output = f"./output/{input_basename}_blurred.mp4"
    OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH", default_output)
    
    # Load filter configuration
    config_values = load_config()
    faceblur_config = FilterFaceblurConfig(**config_values)
    
    print("=" * 60)
    print("Face Blur Filter Pipeline")
    print("=" * 60)
    print(f"Input Video: {VIDEO_INPUT}")
    print(f"Output Video: {OUTPUT_VIDEO_PATH}")
    print(f"Output FPS: {FPS}")
    print(f"Webvis Port: {WEBVIS_PORT}")
    print(f"Detector: {faceblur_config.detector_name}")
    print(f"Blurrer: {faceblur_config.blurrer_name}")
    print(f"Blur Strength: {faceblur_config.blur_strength}")
    print(f"Blur Enabled: {faceblur_config.blur_enabled}")
    print(f"Confidence Threshold: {faceblur_config.detection_confidence_threshold}")
    print(f"Model URL: {faceblur_config.model_artifactory_url}")
    print(f"Model File: {faceblur_config.model_artifact_name}")
    print(f"Debug Mode: {faceblur_config.debug}")
    print("=" * 60)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Define the filter pipeline
    pipeline = [
        # Input video source
        (
            VideoIn,
            {
                "id": "video_in",
                "sources": f"file://{VIDEO_INPUT}!sync!resize=960x540lin",
                "outputs": "tcp://*:5550",
            }
        ),
        
        # Face blur filter
        (
            FilterFaceblur,
            FilterFaceblurConfig(
                id="faceblurfilter",
                sources="tcp://127.0.0.1:5550",
                outputs="tcp://*:5552",
                mq_log="pretty",
                detector_name=faceblur_config.detector_name,
                blurrer_name=faceblur_config.blurrer_name,
                model_artifactory_url=faceblur_config.model_artifactory_url,
                model_artifact_name=faceblur_config.model_artifact_name,
                blur_strength=faceblur_config.blur_strength,
                blur_enabled=faceblur_config.blur_enabled,
                detection_confidence_threshold=faceblur_config.detection_confidence_threshold,
                debug=faceblur_config.debug,
                forward_upstream_data=faceblur_config.forward_upstream_data,
                include_face_coordinates=faceblur_config.include_face_coordinates,
            )
        ),
        # Output video
        (
            VideoOut,
            {
                "id": "video_out",
                "sources": "tcp://127.0.0.1:5552",
                "outputs": f"file://{OUTPUT_VIDEO_PATH}",
            }
        ),
                
        # Web visualization
        (
            Webvis, 
            {
                "id": "webvis", 
                "sources": "tcp://127.0.0.1:5552", 
                "port": WEBVIS_PORT
            }
        )
    ]
    
    print("Starting pipeline...")
    print("Press Ctrl+C to stop")
    print(f"Webvis available at: http://localhost:{WEBVIS_PORT}")
    print("=" * 60)
    
    try:
        # Run the pipeline
        Filter.run_multi(pipeline)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

