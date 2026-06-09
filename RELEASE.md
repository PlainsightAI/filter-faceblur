# Changelog

FaceGuard release notes.

## [Unreleased]

## v1.4.0 - 2026-06-08

### Fixed
- Register `BoxBlur` and `MedianBlur` in the runtime `BLURRERS` registry. The config validator and the platform filter manifest both advertised `blurrer_name` in `['gaussian', 'box', 'median']`, but only `gaussian` was implemented — selecting `box` or `median` crashed `FaceBlur.__init__` with `ValueError: 'box' is not a valid key in the registry`. Both new blurrers mirror `GaussianBlur`'s elliptical-mask + `blur_strength`-scaled kernel pattern.
- Register `HaarDetector` and `DnnDetector` in the runtime `DETECTORS` registry. Same class of bug as the blurrer side: `filter.py` validated `detector_name` against `['yunet', 'haar', 'dnn']` while only `yunet` was implemented, so picking `haar` or `dnn` crashed at construction with the same `ValueError`.
- Clamp every detector's bbox emissions to `[0, w]`/`[0, h]` at the orchestration layer (`FaceBlur.clamp_faces_to_frame`). Detectors can emit boxes extending past frame edges — verified for DNN where SSD outputs are not constrained to `[0, 1]`, and possible for YuNet/Haar on edge faces. Without clamping the blurrer's `image[y:y+h, x:x+w]` slice collapses on a negative start (size==0 early-return) and the detected face is silently left unblurred — a privacy failure for an anonymization filter. The clamp runs in two places: `filter.py.process` right after `detect_faces` so downstream ROI metadata (`face_coordinates`, `face_details`, `meta.detections.rois`) carries in-bounds values, and `FaceBlur.process_frame` as defense-in-depth for direct callers (e.g. `scripts/model_usage.py`). `DnnDetector` also clamps at emit time for the same reason; the orchestration clamp is idempotent.
- `DnnDetector`: wrap network errors during model auto-download as `ValueError` so restricted-egress containers fail clearly instead of leaking a raw `URLError`, matching `YuNetDetector`'s pattern.
- `YuNetDetector._postprocess`: change the confidence filter from `>` to `>=` so detections at exactly the threshold are kept. Aligns with `DnnDetector`'s semantics and matches the natural reading of "minimum confidence". Edge-case behavior change for callers passing detections whose confidence equals the threshold exactly.
- `FaceBlur.process_frame`: thread the accumulating `output` through each blurrer call instead of passing the original `frame` every iteration. Every current blurrer mutates in place so the behavior is unchanged today, but a future copy-returning blurrer would have silently dropped every face except the last.

### Added
- `HaarDetector`: OpenCV-bundled `cv2.CascadeClassifier` frontal-face cascade. No download required — the XML ships with `opencv-python`. Confidence threshold is part of the contract but non-probabilistic; detections are emitted with `confidence=1.0`.
- `DnnDetector`: OpenCV ResNet-SSD Caffe backend (`cv2.dnn.readNetFromCaffe`). Auto-downloads `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel` into the package `weights/` directory on first use; URLs are overridable via `FILTER_DNN_PROTOTXT_URL` and `FILTER_DNN_CAFFEMODEL_URL`.
- Optional SHA-256 verification for model artifacts. When `FILTER_YUNET_SHA256`, `FILTER_DNN_PROTOTXT_SHA256`, or `FILTER_DNN_CAFFEMODEL_SHA256` is set, the downloaded or cached file is hashed and verified before use; mismatch removes the offending file and raises `ValueError` so the next attempt is clean. Unset preserves the existing trust model — opt-in hardening for security-conscious operators.
- Contract test suite for all blurrers (`tests/test_blurrers.py`) and detectors (`tests/test_detectors.py`), plus a registry/regression test (`tests/test_face_blur_registry.py`) that constructs `FaceBlur` against the real `BLURRERS`/`DETECTORS` registries for every advertised name — the existing smoke tests mocked `FaceBlur` and never exercised the lookups, which is why the missing implementations went undetected.

### Changed
- Bump openfilter to `1.1.1` (#15).
- Pin `openfilter-faceblur` to `1.4.0` in `docker-compose.yaml`.
- README: enumerate the supported `FILTER_DETECTOR_NAME` / `FILTER_BLURRER_NAME` values; document the new DNN URL and SHA-256 env vars.
- `model.py` class docstring: list all detector/blurrer options.

## v1.3.0 - 2026-05-21

### Changed
- Bump openfilter to `1.1.0` (#13).
- Update `docker-compose.yaml` and `docker-compose.local.yaml` builtin openfilter images (`openfilter-video-in`, `openfilter-webvis`) to `1.1.0` to match the SDK; pin `openfilter-faceblur` to `1.3.0`.

## v1.2.0 - 2026-05-21

### Changed
- Bump openfilter to `1.0.0` (major SDK release) (#11)
- Update `docker-compose.yaml` and `docker-compose.local.yaml` to the `1.0.0` builtin openfilter images (`openfilter-video-in`, `openfilter-webvis`) and pin `openfilter-faceblur` to `1.2.0`. The local compose file was also pointing at an older `v0.1.10` builtin tag that predates the main compose file's `0.1.27` — both now align on `1.0.0`.

### Fixed
- Make the pre-installed model weights directory writable by `appuser` so `YuNetDetector` can auto-download the face detection model on first run inside the container. The runtime `mkdir` into `/usr/local/lib/python3.11/site-packages/filter_faceblur/model/weights` previously failed under the non-root user with `PermissionError`, crashing `setup()`.
- Rewrite `YuNetDetector._autodownload`: add the missing `return` after the OpenCV download path so execution no longer falls through into the JFrog branch (which double-downloaded the model to a hardcoded `filter_gaf/measurements/model/weights` path left over from a copy-paste), and route both branches through the chown'd `<pkg>/model/weights` dir.
- Drop `shell=True` from the JFrog `curl` invocation in favour of an argv list, eliminating a command-injection vector through interpolated credentials and URL.

### Dependencies
- Bump `python-dotenv` 1.0.1 → 1.2.2 (#4)
- Bump `setuptools` 72.2.0 → 78.1.1 (dev) (#7)
- Bump `pytest` 8.3.4 → 9.0.3 (dev) (#8)
- Bump `wheel` 0.44.0 → 0.46.2 (dev) (#9)

## v1.1.10 - 2026-04-23

### Changed
- Bump openfilter SDK, align CI workflow with shared release gate (source-paths)

- Fix release workflow secret names: `PYPI_API_TOKEN` → `PLAINSIGHT_PYPI_TOKEN`, `DOCKERHUB_TOKEN` → `DOCKERHUB_ACCESS_TOKEN` (org-level secret names). Without this the PyPI / Docker Hub tokens resolved to empty and no package has been published since the migration.
- Bump openfilter dependency to `>=0.1.30`.

## v1.1.9 - 2026-04-23

### Fixed
- Re-release: v1.1.8 tag was created from PR branch (orphan), publish was skipped
- Fix corrupted docker-compose.yaml (broken find-and-replace)
- Update docker-compose builtins to 0.1.27, self-reference to DockerHub

## v1.1.8 - 2026-04-22

### Fixed
- Fix double face detection per frame — removed duplicate detection call that caused each face to appear twice in output

## v1.1.6 - 2026-04-20

### Changed
- Remove redundant ci.yaml (shared workflow handles PR testing)
- Add push + pull_request triggers to create-release.yaml


## v1.1.5 - 2026-04-15

### Changed
- Add CI/CD workflows: create-release.yaml (Docker Hub publishing), ci.yaml (PR testing), security-scan.yaml
- Bump openfilter dependency to >=0.1.27
- Update Makefile IMAGE to Docker Hub path


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
