# Changelog
FaceGuard release notes

## [Unreleased]

## v1.1.4 - 2025-09-27
### Changed
- **Updated Documentation**

## v1.1.0 - 2025-01-16
### Added
- **Upstream Data Forwarding**: Configurable forwarding of data from upstream filters
- **Face Detection Metadata**: Rich face coordinates, confidence scores, and detection details in frame data
- **Environment Variable Configuration**: Complete environment variable support for all parameters
- **Dynamic Output Naming**: Automatic generation of descriptive output video filenames
- **Debug Logging**: Optional debug mode with detailed face detection logging
- **Comprehensive Test Suite**: Integration and smoke tests for all functionality
- **Enhanced Documentation**: Detailed README and overview documentation with usage examples

### Changed
- **Configuration Cleanup**: Removed unused `confidence_threshold` parameter, kept only `detection_confidence_threshold`
- **String-to-Type Conversion**: Improved configuration validation with proper type conversion
- **Usage Script**: Simplified `filter_usage.py` with environment variable support
- **Test Coverage**: Updated all tests to use cleaned configuration parameters

### Fixed
- **Configuration Validation**: Fixed string-to-type conversion issues in `normalize_config`
- **Test Mocking**: Resolved import path issues in test suite
- **Environment Variable Loading**: Implemented proper environment variable handling
- **Documentation**: Updated all documentation to reflect current configuration options

## v1.0.6 - 2025-07-15
### Update
- Migrated from filter_runtime to openfilter

### Added
- Internal improvements: `make install` uses editable mode by default

## v1.0.5 - 2024-03-26

### Added
- Internal improvements

## v1.0.0

### Added
- Initial Release: new filter for detecting and blurring faces in video frames using a pluggable face detection and blur backend.

- **Face Detection**
  - Uses a model (e.g., `face_detection_yunet_2023mar.onnx`) from a configurable remote URL

- **Customizable Detection and Blur Backends**
  - Detector name (`DETECTOR_NAME`) and blurring method (`BLURRER_NAME`) can be set via environment variables

- **Plug-and-Play Processing**
  - Processes the main frame and replaces it with a version where detected faces are blurred

- **Environment Variable Integration**
  - Model location and runtime behavior are controlled through `.env` or environment vars

- **Lightweight Configuration**
  - No required configuration fields; minimal setup needed for integration
