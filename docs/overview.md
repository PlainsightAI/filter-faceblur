---
title: FaceGuard
sidebar_label: Overview
sidebar_position: 1
---

import Admonition from '@theme/Admonition';

# FaceGuard

FaceGuard automatically detects and blurs faces in video frames using state-of-the-art computer vision models. It's designed for privacy-conscious applications that need real-time face anonymization with configurable detection sensitivity and blur intensity.

## How It Works

The filter uses a multi-stage pipeline:

1. **Face Detection**: Uses OpenCV's YuNet model to detect faces in each frame
2. **Confidence Filtering**: Only processes faces above the confidence threshold
3. **Blur Application**: Applies Gaussian blur to detected face regions
4. **Metadata Extraction**: Optionally includes face coordinates and detection confidence
5. **Output Generation**: Returns the processed frame with optional upstream data forwarding

## Features

### **Advanced Face Detection**
- **YuNet Model**: Uses OpenCV's state-of-the-art YuNet face detector
- **Configurable Sensitivity**: Adjust detection confidence threshold (0.0-1.0)
- **Real-time Processing**: Optimized for live video streams
- **Multiple Face Support**: Detects and blurs multiple faces per frame

### **Flexible Configuration**
- **Environment Variable Driven**: No command-line arguments needed
- **Dynamic Output Naming**: Automatically generates descriptive output filenames
- **Debug Mode**: Optional logging for development and troubleshooting
- **Upstream Data Forwarding**: Passes through data from upstream filters

### **Rich Metadata Output**
- **Face Coordinates**: Bounding box and center coordinates for each detected face
- **Confidence Scores**: Detection confidence for each face
- **Face Count**: Total number of faces detected per frame
- **Detection Details**: Structured data for downstream processing

## Configuration Options

### Core Detection Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `FILTER_DETECTOR_NAME` | string | `"yunet"` | Face detector type (`yunet`, `haar`, `dnn`) |
| `FILTER_DETECTION_CONFIDENCE_THRESHOLD` | float | `0.25` | Minimum confidence for face detection (0.0-1.0) |
| `FILTER_BLURRER_NAME` | string | `"gaussian"` | Blur algorithm (`gaussian`, `box`, `median`) |
| `FILTER_BLUR_STRENGTH` | float | `1.0` | Blur intensity (higher = more blur) |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `FILTER_MODEL_ARTIFACTORY_URL` | string | OpenCV Zoo URL | Base URL for model download |
| `FILTER_MODEL_ARTIFACT_NAME` | string | `"face_detection_yunet_2023mar.onnx"` | Model filename |

### Output and Debugging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `FILTER_DEBUG` | boolean | `false` | Enable debug logging |
| `FILTER_FORWARD_UPSTREAM_DATA` | boolean | `true` | Forward data from upstream filters |
| `FILTER_INCLUDE_FACE_COORDINATES` | boolean | `true` | Include face coordinates in frame data |

## Usage Examples

### Basic Usage
```bash
# Run with default settings
python scripts/filter_usage.py
```

### Custom Video Input
```bash
# Process a specific video file
VIDEO_INPUT="./data/surveillance_footage.mp4" python scripts/filter_usage.py
```

### High Sensitivity Detection
```bash
# Detect more faces (lower confidence threshold)
FILTER_DETECTION_CONFIDENCE_THRESHOLD=0.1 python scripts/filter_usage.py
```

### Strong Blur Effect
```bash
# Apply heavy blurring
FILTER_BLUR_STRENGTH=3.0 python scripts/filter_usage.py
```

### Debug Mode
```bash
# Enable debug logging
FILTER_DEBUG=true python scripts/filter_usage.py
```

## Output Data Structure

The filter enriches each frame with detailed face detection metadata:

```json
{
  "faces_detected": 2,
  "face_coordinates": [
    {
      "bbox": [100, 50, 80, 100],
      "confidence": 0.95
    }
  ],
  "face_details": [
    {
      "face_id": 0,
      "bounding_box": {
        "x": 100, "y": 50,
        "width": 80, "height": 100
      },
      "center": {
        "x": 140, "y": 100
      },
      "confidence": 0.95
    }
  ]
}
```

## When to Use

### **Perfect For:**
- **Surveillance Systems**: Anonymize faces in security footage
- **Retail Analytics**: Protect customer privacy in store monitoring
- **Live Streaming**: Real-time face blurring for privacy compliance
- **Social Media**: Automatic face anonymization for content sharing
- **Research**: Privacy-preserving video analysis

### **Consider Alternatives For:**
- **High-Resolution Processing**: Very large images may impact performance
- **Batch Processing**: Designed for real-time streams, not batch operations
- **Custom Detection**: Limited to face detection (not objects or other features)

## Performance Tuning

### Detection Sensitivity
- **Lower threshold (0.1-0.2)**: More faces detected, higher false positives
- **Higher threshold (0.5-0.8)**: Fewer faces detected, higher precision
- **Default (0.25)**: Balanced approach for most use cases

### Blur Strength
- **Light blur (0.5-1.0)**: Faces still somewhat recognizable
- **Medium blur (1.0-2.0)**: Good privacy protection (default)
- **Heavy blur (2.0+)**: Complete anonymization

<Admonition type="tip" title="Pro Tip">
For best results, test different confidence thresholds with your specific video content to find the optimal balance between detection accuracy and false positives.
</Admonition>

<Admonition type="warning" title="Important">
Ensure the model file is publicly accessible or pre-downloaded for edge deployments. The filter will automatically download models from the configured URL on first run.
</Admonition>