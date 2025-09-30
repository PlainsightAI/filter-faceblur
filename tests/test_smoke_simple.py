"""
Smoke tests for FilterFaceblur basic functionality.

These tests verify that the filter can be initialized, configured, and perform
basic operations without complex pipeline orchestration.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from openfilter.filter_runtime.filter import Frame
from filter_faceblur.filter import FilterFaceblur, FilterFaceblurConfig


class TestSmokeSimple:
    """Test basic filter functionality and lifecycle."""

    @pytest.fixture
    def temp_workdir(self):
        """Create a temporary working directory for tests."""
        with tempfile.TemporaryDirectory(prefix='filter_faceblur_smoke_') as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        frame_data = {"meta": {"id": 1, "topic": "test"}}
        return Frame(image, frame_data, 'BGR')

    @pytest.fixture
    def mock_face_blur(self):
        """Create a mock FaceBlur object for testing."""
        mock_face_blur = MagicMock()
        mock_face_blur.process_frame.return_value = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        return mock_face_blur

    def test_filter_initialization(self, temp_workdir):
        """Test that the filter can be initialized with valid config."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.5,
            'detection_confidence_threshold': 0.7,
            'debug': False
        }
        
        # Test config normalization
        config = FilterFaceblur.normalize_config(config_data)
        assert config.detector_name == 'yunet'
        assert config.blurrer_name == 'gaussian'
        assert config.blur_strength == 1.5
        assert config.detection_confidence_threshold == 0.7
        assert config.debug is False
        
        # Test filter initialization
        filter_instance = FilterFaceblur(config=config)
        assert filter_instance is not None

    def test_setup_and_shutdown(self, temp_workdir, mock_face_blur):
        """Test that setup() and shutdown() work correctly."""
        config_data = {
            'detector_name': 'haar',
            'blurrer_name': 'box',
            'blur_strength': 2.0,
            'detection_confidence_threshold': 0.8,
            'debug': True
        }
        
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
        
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
            
            # Test setup
            filter_instance.setup(config)
            assert filter_instance.config is not None
            assert filter_instance.face_blur is not None
            assert filter_instance.blur_strength == 2.0
            assert filter_instance.detection_confidence_threshold == 0.8
            assert filter_instance.debug is True
            
            # Test shutdown
            filter_instance.shutdown()  # Should not raise any exceptions

    def test_config_validation(self):
        """Test that configuration validation works correctly."""
        # Test valid configuration
        valid_config = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5
        }
        
        config = FilterFaceblur.normalize_config(valid_config)
        assert config.detector_name == 'yunet'
        assert config.blurrer_name == 'gaussian'
        
        # Test configuration with typo - should not raise error anymore
        config_with_typo = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'detector_names': 'yunet'  # Typo
        }
        
        # Should not raise an error - unknown keys are passed through
        config = FilterFaceblur.normalize_config(config_with_typo)
        assert config.detector_name == 'yunet'
        assert config.blurrer_name == 'gaussian'

    def test_frame_processing(self, sample_frame, mock_face_blur):
        """Test basic frame processing."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
        
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
        
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
            
            filter_instance.setup(config)
            
            # Process frame
            frames = {"main": sample_frame}
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert "main" in output_frames
            assert len(output_frames) == 1
            assert isinstance(output_frames["main"], Frame)
            
            # Verify FaceBlur was called
            mock_face_blur.process_frame.assert_called_once()

    def test_empty_frame_processing(self, mock_face_blur):
        """Test processing with empty frame dictionary."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
        
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
        
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
            
            filter_instance.setup(config)
            
            # Process empty frame
            frames = {}
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert len(output_frames) == 0
            
            # Verify FaceBlur was not called
            mock_face_blur.process_frame.assert_not_called()

    def test_missing_main_frame(self, sample_frame, mock_face_blur):
        """Test processing when main frame is missing but other topics have images."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
    
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
    
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
    
            filter_instance.setup(config)
    
            # Process frame without main key but with other topic containing image
            frames = {"other": sample_frame}
            output_frames = filter_instance.process(frames)
    
            # Verify output (should process the other topic since it contains an image)
            assert "other" in output_frames
            assert len(output_frames) == 1
    
            # Verify FaceBlur was called for the other topic
            mock_face_blur.process_frame.assert_called_once()
            mock_face_blur.detector.detect_faces.assert_called_once()

    def test_multiple_topics_processing(self, sample_frame, mock_face_blur):
        """Test processing multiple topics with images."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
    
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
    
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
    
            filter_instance.setup(config)
    
            # Process frames with multiple topics containing images
            frames = {
                "main": sample_frame,
                "stream2": sample_frame,
                "data_only": Frame({"some": "data"})  # Data-only frame
            }
            output_frames = filter_instance.process(frames)
    
            # Verify output (should process both image topics and forward data-only frame)
            assert "main" in output_frames
            assert "stream2" in output_frames
            assert "data_only" in output_frames
            assert len(output_frames) == 3
    
            # Verify FaceBlur was called twice (once for each image topic)
            assert mock_face_blur.process_frame.call_count == 2
            assert mock_face_blur.detector.detect_faces.call_count == 2

    def test_blur_disabled_processing(self, sample_frame, mock_face_blur):
        """Test processing when blur is disabled but faces are still detected."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'blur_enabled': False,  # Disable blurring
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
    
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
    
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
    
            filter_instance.setup(config)
    
            # Process frame
            frames = {"main": sample_frame}
            output_frames = filter_instance.process(frames)
    
            # Verify output
            assert "main" in output_frames
            assert len(output_frames) == 1
    
            # Verify face detection was called but blurring was not
            mock_face_blur.detector.detect_faces.assert_called_once()
            mock_face_blur.process_frame.assert_not_called()

    def test_blur_strength_zero_processing(self, sample_frame, mock_face_blur):
        """Test processing when blur strength is 0 but faces are still detected."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 0.0,  # Zero blur strength
            'blur_enabled': True,
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
    
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
    
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
    
            filter_instance.setup(config)
    
            # Process frame
            frames = {"main": sample_frame}
            output_frames = filter_instance.process(frames)
    
            # Verify output
            assert "main" in output_frames
            assert len(output_frames) == 1
    
            # Verify face detection was called but blurring was not
            mock_face_blur.detector.detect_faces.assert_called_once()
            mock_face_blur.process_frame.assert_not_called()

    def test_main_topic_ordering(self, sample_frame, mock_face_blur):
        """Test that main topic always comes first in output dictionary."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'blur_enabled': True,
            'detection_confidence_threshold': 0.5,
            'forward_upstream_data': True,
            'debug': True
        }
    
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
    
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
    
            filter_instance.setup(config)
    
            # Process frames with multiple topics in different order
            frames = {
                "stream2": sample_frame,
                "main": sample_frame,
                "other": sample_frame
            }
            output_frames = filter_instance.process(frames)
    
            # Verify main topic comes first regardless of input order
            output_keys = list(output_frames.keys())
            assert output_keys[0] == "main"
            assert len(output_frames) == 3  # All topics processed

    def test_different_detector_types(self, sample_frame, mock_face_blur):
        """Test processing with different detector types."""
        detectors = ['yunet', 'haar', 'dnn']
        
        for detector in detectors:
            config_data = {
                'detector_name': detector,
                'blurrer_name': 'gaussian',
                'blur_strength': 1.0,
                'detection_confidence_threshold': 0.5,
                'debug': False
            }
            
            config = FilterFaceblur.normalize_config(config_data)
            filter_instance = FilterFaceblur(config=config)
            
            # Mock the FaceBlur import and initialization
            with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
                mock_face_blur_class.return_value = mock_face_blur
                
                filter_instance.setup(config)
                
                # Process frame
                frames = {"main": sample_frame}
                output_frames = filter_instance.process(frames)
                
                # Verify output
                assert "main" in output_frames
                assert len(output_frames) == 1

    def test_different_blurrer_types(self, sample_frame, mock_face_blur):
        """Test processing with different blurrer types."""
        blurrers = ['gaussian', 'box', 'median']
        
        for blurrer in blurrers:
            config_data = {
                'detector_name': 'yunet',
                'blurrer_name': blurrer,
                'blur_strength': 1.0,
                'detection_confidence_threshold': 0.5,
                'debug': False
            }
            
            config = FilterFaceblur.normalize_config(config_data)
            filter_instance = FilterFaceblur(config=config)
            
            # Mock the FaceBlur import and initialization
            with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
                mock_face_blur_class.return_value = mock_face_blur
                
                filter_instance.setup(config)
                
                # Process frame
                frames = {"main": sample_frame}
                output_frames = filter_instance.process(frames)
                
                # Verify output
                assert "main" in output_frames
                assert len(output_frames) == 1

    def test_debug_mode_processing(self, sample_frame, mock_face_blur):
        """Test processing with debug mode enabled."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5,
            'debug': True
        }
        
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
        
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
            
            filter_instance.setup(config)
            
            # Process frame with debug enabled
            frames = {"main": sample_frame}
            
            with patch('filter_faceblur.filter.logger') as mock_logger:
                output_frames = filter_instance.process(frames)
                
                # Verify debug logging was called
                assert mock_logger.info.called

    def test_string_config_conversion(self):
        """Test that string configs are properly converted to types."""
        # Test with string values that should be converted
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': '2.5',
            'detection_confidence_threshold': '0.8',
            'debug': 'true'
        }
        
        normalized = FilterFaceblur.normalize_config(config_data)
        
        # Check that string values are converted to correct types
        assert normalized.detector_name == 'yunet'
        assert normalized.blurrer_name == 'gaussian'
        assert normalized.blur_strength == 2.5
        assert normalized.detection_confidence_threshold == 0.8
        assert normalized.debug is True

    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configuration values."""
        # Test invalid detector name
        config_invalid_detector = {
            'detector_name': 'invalid_detector',
            'blurrer_name': 'gaussian'
        }
        
        with pytest.raises(ValueError, match="Invalid detector_name"):
            FilterFaceblur.normalize_config(config_invalid_detector)
        
        # Test invalid blurrer name
        config_invalid_blurrer = {
            'detector_name': 'yunet',
            'blurrer_name': 'invalid_blurrer'
        }
        
        with pytest.raises(ValueError, match="Invalid blurrer_name"):
            FilterFaceblur.normalize_config(config_invalid_blurrer)
        
        # Test invalid blur strength
        config_invalid_strength = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'blur_strength': -1.0
        }
        
        with pytest.raises(ValueError, match="Blur strength must be non-negative"):
            FilterFaceblur.normalize_config(config_invalid_strength)
        
        # Test invalid confidence threshold
        config_invalid_threshold = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'detection_confidence_threshold': 1.5
        }
        
        with pytest.raises(ValueError, match="Detection confidence threshold must be between 0 and 1"):
            FilterFaceblur.normalize_config(config_invalid_threshold)

    def test_environment_variable_loading(self):
        """Test environment variable configuration loading."""
        # Set environment variables
        os.environ['FILTER_DETECTOR_NAME'] = 'haar'
        os.environ['FILTER_BLURRER_NAME'] = 'box'
        os.environ['FILTER_BLUR_STRENGTH'] = '3.0'
        os.environ['FILTER_DETECTION_CONFIDENCE_THRESHOLD'] = '0.9'
        os.environ['FILTER_DEBUG'] = 'true'
        
        try:
            config = FilterFaceblurConfig()
            normalized = FilterFaceblur.normalize_config(config)
            
            assert normalized.detector_name == 'haar'
            assert normalized.blurrer_name == 'box'
            assert normalized.blur_strength == 3.0
            assert normalized.detection_confidence_threshold == 0.9
            assert normalized.debug is True
            
        finally:
            # Clean up environment variables
            for key in ['FILTER_DETECTOR_NAME', 'FILTER_BLURRER_NAME', 'FILTER_BLUR_STRENGTH', 
                       'FILTER_CONFIDENCE_THRESHOLD', 'FILTER_DEBUG']:
                if key in os.environ:
                    del os.environ[key]

    def test_model_artifact_configuration(self, sample_frame, mock_face_blur):
        """Test processing with custom model artifact configuration."""
        config_data = {
            'detector_name': 'yunet',
            'blurrer_name': 'gaussian',
            'model_artifactory_url': 'https://example.com/models/',
            'model_artifact_name': 'custom_model.onnx',
            'blur_strength': 1.0,
            'detection_confidence_threshold': 0.5,
            'debug': False
        }
        
        config = FilterFaceblur.normalize_config(config_data)
        filter_instance = FilterFaceblur(config=config)
        
        # Mock the FaceBlur import and initialization
        with patch('filter_faceblur.model.FaceBlur') as mock_face_blur_class:
            mock_face_blur_class.return_value = mock_face_blur
            
            filter_instance.setup(config)
            
            # Verify FaceBlur was initialized with custom model
            mock_face_blur_class.assert_called_once()
            call_args = mock_face_blur_class.call_args
            assert 'https://example.com/models/custom_model.onnx' in str(call_args)
            
            # Process frame
            frames = {"main": sample_frame}
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert "main" in output_frames
            assert len(output_frames) == 1


if __name__ == '__main__':
    pytest.main([__file__])
