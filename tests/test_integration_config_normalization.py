"""
Integration tests for FilterFaceblur configuration normalization.

These tests verify that the normalize_config method properly handles various
configuration inputs, validates parameters, and provides helpful error messages.
"""

import pytest
import os
from filter_faceblur.filter import FilterFaceblur, FilterFaceblurConfig


class TestIntegrationConfigNormalization:
    """Test comprehensive configuration normalization scenarios."""

    def test_string_to_type_conversions(self):
        """Test that string configurations are properly converted to correct types."""
        
        config_with_string_values = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'model_artifactory_url': 'https://example.com/models/',
            'model_artifact_name': 'model.onnx',
            'blur_strength': '2.5',  # String float
            'detection_confidence_threshold': '0.7',  # String float
            'debug': 'true'  # String bool
        }
        
        normalized = FilterFaceblur.normalize_config(config_with_string_values)
        
        # Check that string values are preserved or converted correctly
        assert isinstance(normalized.detector_name, str)
        assert normalized.detector_name == 'yunet'
        assert isinstance(normalized.blurrer_name, str)
        assert normalized.blurrer_name == 'gaussian'
        assert isinstance(normalized.model_artifactory_url, str)
        assert normalized.model_artifactory_url == 'https://example.com/models/'
        assert isinstance(normalized.model_artifact_name, str)
        assert normalized.model_artifact_name == 'model.onnx'
        assert isinstance(normalized.blur_strength, float)
        assert normalized.blur_strength == 2.5
        assert isinstance(normalized.detection_confidence_threshold, float)
        assert normalized.detection_confidence_threshold == 0.7
        assert isinstance(normalized.debug, bool)
        assert normalized.debug is True

    def test_required_vs_optional_parameters(self):
        """Test that required parameters are validated correctly."""
        
        # Test minimal valid configuration
        minimal_config = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian'
        }
        
        normalized = FilterFaceblur.normalize_config(minimal_config)
        assert normalized.detector_name == 'yunet'
        assert normalized.blurrer_name == 'gaussian'
        assert normalized.model_artifactory_url == "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"  # Default value
        assert normalized.model_artifact_name == "face_detection_yunet_2023mar.onnx"  # Default value
        assert normalized.blur_strength == 1.0  # Default value
        assert normalized.detection_confidence_threshold == 0.25  # Default value
        assert normalized.debug is False  # Default value

    def test_detector_name_validation(self):
        """Test detector name validation."""
        
        # Test valid detector names
        valid_detectors = ['yunet', 'haar', 'dnn']
        
        for detector in valid_detectors:
            config = {
                'detector_name': detector,
                'blurrer_name': 'gaussian'
            }
            normalized = FilterFaceblur.normalize_config(config)
            assert normalized.detector_name == detector
        
        # Test invalid detector name
        invalid_config = {
            'detector_name': 'invalid_detector',
            'blurrer_name': 'gaussian'
        }
        
        with pytest.raises(ValueError, match="Invalid detector_name"):
            FilterFaceblur.normalize_config(invalid_config)

    def test_blurrer_name_validation(self):
        """Test blurrer name validation."""
        
        # Test valid blurrer names
        valid_blurrers = ['gaussian', 'box', 'median']
        
        for blurrer in valid_blurrers:
            config = {
                'detector_name': 'yunet',
                'blurrer_name': blurrer
            }
            normalized = FilterFaceblur.normalize_config(config)
            assert normalized.blurrer_name == blurrer
        
        # Test invalid blurrer name
        invalid_config = {
            'detector_name': 'yunet',
            'blurrer_name': 'invalid_blurrer'
        }
        
        with pytest.raises(ValueError, match="Invalid blurrer_name"):
            FilterFaceblur.normalize_config(invalid_config)

    def test_blur_strength_validation(self):
        """Test blur strength validation."""
        
        # Test valid blur strengths
        valid_strengths = [0.1, 1.0, 2.5, 10.0]
        
        for strength in valid_strengths:
            config = {
                'detector_name': 'yunet',
                'blurrer_name': 'gaussian',
                'blur_strength': strength
            }
            normalized = FilterFaceblur.normalize_config(config)
            assert normalized.blur_strength == strength
        
        # Test invalid blur strength (negative)
        config_negative = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': -1.0
        }
        
        with pytest.raises(ValueError, match="Blur strength must be non-negative"):
            FilterFaceblur.normalize_config(config_negative)

    def test_detection_confidence_threshold_validation(self):
        """Test detection confidence threshold validation."""
        
        # Test valid confidence thresholds
        valid_thresholds = [0.0, 0.25, 0.5, 0.7, 1.0]
        
        for threshold in valid_thresholds:
            config = {
                'detector_name': 'yunet',
                'blurrer_name': 'gaussian',
                'detection_confidence_threshold': threshold
            }
            normalized = FilterFaceblur.normalize_config(config)
            assert normalized.detection_confidence_threshold == threshold
        
        # Test invalid confidence thresholds
        invalid_thresholds = [-0.1, 1.1, 2.0]
        
        for threshold in invalid_thresholds:
            config = {
                'detector_name': 'yunet',
                'blurrer_name': 'gaussian',
                'detection_confidence_threshold': threshold
            }
            with pytest.raises(ValueError, match="Detection confidence threshold must be between 0 and 1"):
                FilterFaceblur.normalize_config(config)

    def test_debug_mode_configuration(self):
        """Test debug mode configuration options."""
        
        # Test debug = True
        config_debug_true = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'debug': True
        }
        
        normalized = FilterFaceblur.normalize_config(config_debug_true)
        assert normalized.debug is True
        
        # Test debug = False
        config_debug_false = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'debug': False
        }
        
        normalized = FilterFaceblur.normalize_config(config_debug_false)
        assert normalized.debug is False
        
        # Test debug = "true" (string)
        config_debug_string_true = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'debug': "true"
        }
        
        normalized = FilterFaceblur.normalize_config(config_debug_string_true)
        assert normalized.debug is True
        
        # Test debug = "false" (string)
        config_debug_string_false = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'debug': "false"
        }
        
        normalized = FilterFaceblur.normalize_config(config_debug_string_false)
        assert normalized.debug is False
        
        # Test invalid debug value
        config_invalid_debug = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'debug': "maybe"
        }
        
        with pytest.raises(ValueError, match="Invalid debug mode"):
            FilterFaceblur.normalize_config(config_invalid_debug)

    def test_blur_enabled_validation(self):
        """Test blur_enabled parameter validation."""
        # Test valid blur_enabled values
        for blur_enabled in [True, False, 'true', 'false', 'True', 'False']:
            config = {'blur_enabled': blur_enabled}
            normalized = FilterFaceblur.normalize_config(config)
            expected = blur_enabled.lower() == 'true' if isinstance(blur_enabled, str) else blur_enabled
            assert normalized.blur_enabled == expected
        
        # Test invalid blur_enabled string values (these will be converted to bool in the string processing)
        for invalid_value in ['invalid', 'yes', 'no']:
            config = {'blur_enabled': invalid_value}
            with pytest.raises(ValueError, match="Invalid blur_enabled mode"):
                FilterFaceblur.normalize_config(config)
        
        # Test invalid non-string values (these will fail the isinstance check)
        for invalid_value in [123, None, 1.5]:
            config = {'blur_enabled': invalid_value}
            with pytest.raises(ValueError, match="Invalid blur_enabled mode"):
                FilterFaceblur.normalize_config(config)

    def test_environment_variable_loading(self):
        """Test environment variable configuration loading."""
        
        # Set environment variables
        os.environ['FILTER_DETECTOR_NAME'] = 'haar'
        os.environ['FILTER_BLURRER_NAME'] = 'box'
        os.environ['FILTER_MODEL_ARTIFACTORY_URL'] = 'https://example.com/models/'
        os.environ['FILTER_MODEL_ARTIFACT_NAME'] = 'custom_model.onnx'
        os.environ['FILTER_BLUR_STRENGTH'] = '3.0'
        os.environ['FILTER_DETECTION_CONFIDENCE_THRESHOLD'] = '0.8'
        os.environ['FILTER_DEBUG'] = 'true'
        
        try:
            config = FilterFaceblurConfig()
            normalized = FilterFaceblur.normalize_config(config)
            
            assert normalized.detector_name == 'haar'
            assert normalized.blurrer_name == 'box'
            assert normalized.model_artifactory_url == 'https://example.com/models/'
            assert normalized.model_artifact_name == 'custom_model.onnx'
            assert normalized.blur_strength == 3.0
            assert normalized.detection_confidence_threshold == 0.8
            assert normalized.debug is True
            
        finally:
            # Clean up environment variables
            for key in ['FILTER_DETECTOR_NAME', 'FILTER_BLURRER_NAME', 'FILTER_MODEL_ARTIFACTORY_URL', 
                       'FILTER_MODEL_ARTIFACT_NAME', 'FILTER_BLUR_STRENGTH', 'FILTER_CONFIDENCE_THRESHOLD', 'FILTER_DEBUG']:
                if key in os.environ:
                    del os.environ[key]

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        
        # Test very small positive values
        config_small_values = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 0.001,
            'detection_confidence_threshold': 0.001
        }
        
        normalized = FilterFaceblur.normalize_config(config_small_values)
        assert normalized.blur_strength == 0.001
        assert normalized.detection_confidence_threshold == 0.001
        
        # Test very large values
        config_large_values = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1000.0,
            'detection_confidence_threshold': 1.0
        }
        
        normalized = FilterFaceblur.normalize_config(config_large_values)
        assert normalized.blur_strength == 1000.0
        assert normalized.detection_confidence_threshold == 1.0
        
        # Test empty string values
        config_empty_strings = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'model_artifactory_url': '',  # Empty string should be allowed
            'model_artifact_name': ''     # Empty string should be allowed
        }
        
        normalized = FilterFaceblur.normalize_config(config_empty_strings)
        assert normalized.model_artifactory_url == ''
        assert normalized.model_artifact_name == ''

    def test_unknown_config_key_validation(self):
        """Test that unknown configuration keys are handled gracefully."""
        
        # Test with a typo in a common parameter - should not raise error anymore
        config_with_typo = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'detector_names': 'yunet'  # Typo: should be 'detector_name'
        }
        
        # Should not raise an error - unknown keys are passed through
        config = FilterFaceblur.normalize_config(config_with_typo)
        assert config.detector_name == 'yunet'
        assert config.blurrer_name == 'gaussian'
        
        # Test with completely unknown key - should not raise error
        config_unknown = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'unknown_parameter': 'value'
        }
        
        # Should not raise an error - unknown keys are passed through
        config = FilterFaceblur.normalize_config(config_unknown)
        assert config.detector_name == 'yunet'
        assert config.blurrer_name == 'gaussian'

    def test_runtime_keys_ignored(self):
        """Test that OpenFilter runtime keys are ignored during validation."""
        
        # Test with runtime keys that should be ignored
        config_with_runtime_keys = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'pipeline_id': 'test_pipeline',  # Runtime key
            'device_name': 'test_device',    # Runtime key
            'log_path': '/tmp/logs',         # Runtime key
            'id': 'test_filter',             # Runtime key
            'sources': 'tcp://localhost:5550',  # Runtime key
            'outputs': 'tcp://localhost:5551',  # Runtime key
            'workdir': '/tmp/work'           # Runtime key
        }
        
        # Should not raise an error
        normalized = FilterFaceblur.normalize_config(config_with_runtime_keys)
        assert normalized.detector_name == 'yunet'
        assert normalized.blurrer_name == 'gaussian'

    def test_comprehensive_configuration(self):
        """Test a comprehensive configuration with all parameters."""
        
        comprehensive_config = {
            'detector_name': 'haar',
            'blurrer_name': 'box',
            'model_artifactory_url': 'https://example.com/models/',
            'model_artifact_name': 'custom_model.onnx',
            'blur_strength': 2.5,
            'blur_enabled': False,
            'detection_confidence_threshold': 0.8,
            'debug': True
        }
        
        normalized = FilterFaceblur.normalize_config(comprehensive_config)
        
        # Verify all parameters are correctly set
        assert normalized.detector_name == 'haar'
        assert normalized.blurrer_name == 'box'
        assert normalized.model_artifactory_url == 'https://example.com/models/'
        assert normalized.model_artifact_name == 'custom_model.onnx'
        assert normalized.blur_strength == 2.5
        assert normalized.blur_enabled == False
        assert normalized.detection_confidence_threshold == 0.8
        assert normalized.debug is True
