# FaceGuard

[![PyPI version](https://img.shields.io/pypi/v/filter-faceblur.svg?style=flat-square)](https://pypi.org/project/filter-faceblur/)
[![Docker Version](https://img.shields.io/docker/v/plainsightai/openfilter-faceblur?sort=semver)](https://hub.docker.com/r/plainsightai/openfilter-faceblur)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PlainsightAI/filter-faceblur/blob/main/LICENSE)

FaceGuard is a computer vision filter that automatically detects and blurs faces in video streams using OpenCV's YuNet face detection model. Perfect for privacy-conscious applications that need real-time face anonymization.

## Demo

Here are some examples of FaceGuard in action:

![Face Blur Demo 1](https://github.com/PlainsightAI/filter-faceblur/blob/main/assets/video-01_blurred.gif)
![Face Blur Demo 2](https://github.com/PlainsightAI/filter-faceblur/blob/main/assets/video-02_blurred.gif)
![Face Blur Demo 3](https://github.com/PlainsightAI/filter-faceblur/blob/main/assets/video-03_blurred.gif)

## Quick Start

The easiest way to run FaceGuard is using the provided usage script:

```bash
# Basic usage with default settings
python scripts/filter_usage.py

# Custom video input
VIDEO_INPUT="./data/your-video.mp4" python scripts/filter_usage.py

# Custom configuration
FILTER_DETECTION_CONFIDENCE_THRESHOLD=0.3 FILTER_BLUR_STRENGTH=2.0 python scripts/filter_usage.py
```

### Environment Variables

The filter can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_INPUT` | `./data/video-01.mp4` | Input video file path |
| `OUTPUT_VIDEO_PATH` | `./output/{input_name}_blurred.mp4` | Output video file path |
| `OUTPUT_FPS` | `30` | Output video frames per second |
| `WEBVIS_PORT` | `8000` | Port for Webvis visualization |
| `FILTER_DETECTOR_NAME` | `yunet` | Face detector type |
| `FILTER_BLURRER_NAME` | `gaussian` | Blur algorithm type |
| `FILTER_BLUR_STRENGTH` | `1.0` | Blur intensity |
| `FILTER_DETECTION_CONFIDENCE_THRESHOLD` | `0.25` | Minimum confidence for face detection |
| `FILTER_DEBUG` | `False` | Enable debug logging |
| `FILTER_FORWARD_UPSTREAM_DATA` | `True` | Forward data from upstream filters |
| `FILTER_INCLUDE_FACE_COORDINATES` | `True` | Include face coordinates in frame data |

### Viewing Results

After running the filter, you can view the results at:
- **Webvis**: `http://localhost:8000` - Real-time video stream
- **Output Video**: Check the `./output/` directory for the processed video file

## Documentation

For detailed information about configuration options, performance tuning, and advanced usage, see the [comprehensive documentation](https://github.com/PlainsightAI/filter-faceblur/blob/main/docs/overview.md).

## Features

- **Real-time Face Detection**: Uses OpenCV's YuNet model for accurate face detection
- **Configurable Blurring**: Adjustable blur strength and detection sensitivity
- **Rich Metadata**: Face coordinates, confidence scores, and detection details
- **Environment Variable Configuration**: No command-line arguments needed
- **Upstream Data Forwarding**: Passes through data from other filters
- **Debug Mode**: Optional logging for development and troubleshooting

## Install

To install the filter and its dependencies:

```bash
# Create and activate virtual environment
virtualenv venv
source venv/bin/activate

# Install the filter
make install
```

## Run locally

To run the filter locally:

```bash
make run
```

Then navigate to `http://localhost:8000` to see the video stream.

## Run in Docker

Build and run the filter in Docker:

```bash
# Build the Docker image
make build-image

# Run the filter
make run-image
```

Navigate to `http://localhost:8000` to view the video stream.

## Testing

Run the test suite:

```bash
make test
```