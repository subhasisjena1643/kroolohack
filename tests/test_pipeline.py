"""
Test Suite for Real-time Classroom Engagement Analyzer
Performance and functionality tests
"""

import unittest
import time
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import config
from src.modules.face_detector import FaceDetector
from src.modules.pose_estimator import HeadPoseEstimator
from src.modules.gesture_recognizer import GestureRecognizer
from src.modules.audio_processor import AudioProcessor
from src.modules.engagement_scorer import EngagementScorer
from src.utils.communication import CommunicationManager

class TestPerformance(unittest.TestCase):
    """Performance tests for the engagement analyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.target_fps = 30
        self.max_latency_ms = 100  # 100ms max latency
    
    def test_face_detection_performance(self):
        """Test face detection performance"""
        print("\n🧪 Testing Face Detection Performance...")
        
        face_detector = FaceDetector(config.video.__dict__)
        self.assertTrue(face_detector.initialize())
        
        # Warm up
        for _ in range(5):
            face_detector.process_data(self.test_frame)
        
        # Performance test
        start_time = time.time()
        num_frames = 100
        
        for _ in range(num_frames):
            result = face_detector.process_data(self.test_frame)
            self.assertIsInstance(result, dict)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_frames
        fps = 1.0 / avg_time
        
        print(f"   Face Detection FPS: {fps:.1f}")
        print(f"   Average processing time: {avg_time*1000:.1f}ms")
        
        # Assert performance requirements
        self.assertGreater(fps, self.target_fps * 0.8)  # At least 80% of target FPS
        self.assertLess(avg_time * 1000, self.max_latency_ms)
        
        face_detector.cleanup()
    
    def test_pose_estimation_performance(self):
        """Test pose estimation performance"""
        print("\n🧪 Testing Pose Estimation Performance...")
        
        pose_estimator = HeadPoseEstimator(config.video.__dict__)
        self.assertTrue(pose_estimator.initialize())
        
        # Create test data
        test_data = {'frame': self.test_frame, 'faces': []}
        
        # Warm up
        for _ in range(5):
            pose_estimator.process_data(test_data)
        
        # Performance test
        start_time = time.time()
        num_frames = 50  # Pose estimation is more expensive
        
        for _ in range(num_frames):
            result = pose_estimator.process_data(test_data)
            self.assertIsInstance(result, dict)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_frames
        fps = 1.0 / avg_time
        
        print(f"   Pose Estimation FPS: {fps:.1f}")
        print(f"   Average processing time: {avg_time*1000:.1f}ms")
        
        # More lenient requirements for pose estimation
        self.assertGreater(fps, self.target_fps * 0.5)  # At least 50% of target FPS
        self.assertLess(avg_time * 1000, self.max_latency_ms * 2)
        
        pose_estimator.cleanup()
    
    def test_gesture_recognition_performance(self):
        """Test gesture recognition performance"""
        print("\n🧪 Testing Gesture Recognition Performance...")
        
        gesture_recognizer = GestureRecognizer(config.video.__dict__)
        self.assertTrue(gesture_recognizer.initialize())
        
        # Warm up
        for _ in range(5):
            gesture_recognizer.process_data(self.test_frame)
        
        # Performance test
        start_time = time.time()
        num_frames = 50
        
        for _ in range(num_frames):
            result = gesture_recognizer.process_data(self.test_frame)
            self.assertIsInstance(result, dict)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_frames
        fps = 1.0 / avg_time
        
        print(f"   Gesture Recognition FPS: {fps:.1f}")
        print(f"   Average processing time: {avg_time*1000:.1f}ms")
        
        self.assertGreater(fps, self.target_fps * 0.5)
        self.assertLess(avg_time * 1000, self.max_latency_ms * 2)
        
        gesture_recognizer.cleanup()
    
    def test_engagement_scoring_performance(self):
        """Test engagement scoring performance"""
        print("\n🧪 Testing Engagement Scoring Performance...")
        
        engagement_scorer = EngagementScorer(config.engagement.__dict__)
        self.assertTrue(engagement_scorer.initialize())
        
        # Create test data
        test_data = {
            'face_detection': {'face_count': 5, 'attendance_count': 10},
            'pose_estimation': {'average_attention': 0.7, 'attention_distribution': {'high': 0.6, 'medium': 0.3, 'low': 0.1}},
            'gesture_recognition': {'participation_score': 0.5, 'participation_events': []},
            'audio_processing': {'engagement_metrics': {'audio_engagement_score': 0.6}}
        }
        
        # Performance test
        start_time = time.time()
        num_iterations = 1000
        
        for _ in range(num_iterations):
            result = engagement_scorer.process_data(test_data)
            self.assertIsInstance(result, dict)
            self.assertIn('overall_engagement_score', result)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        print(f"   Engagement Scoring: {avg_time*1000:.2f}ms per calculation")
        
        # Engagement scoring should be very fast
        self.assertLess(avg_time * 1000, 5)  # Less than 5ms
        
        engagement_scorer.cleanup()
    
    def test_memory_usage(self):
        """Test memory usage of components"""
        print("\n🧪 Testing Memory Usage...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize all components
        face_detector = FaceDetector(config.video.__dict__)
        face_detector.initialize()
        
        pose_estimator = HeadPoseEstimator(config.video.__dict__)
        pose_estimator.initialize()
        
        gesture_recognizer = GestureRecognizer(config.video.__dict__)
        gesture_recognizer.initialize()
        
        engagement_scorer = EngagementScorer(config.engagement.__dict__)
        engagement_scorer.initialize()
        
        # Process some frames
        for _ in range(100):
            face_result = face_detector.process_data(self.test_frame)
            pose_result = pose_estimator.process_data({'frame': self.test_frame, 'faces': []})
            gesture_result = gesture_recognizer.process_data(self.test_frame)
            
            combined_data = {
                'face_detection': face_result,
                'pose_estimation': pose_result,
                'gesture_recognition': gesture_result,
                'audio_processing': {'engagement_metrics': {'audio_engagement_score': 0.5}}
            }
            engagement_scorer.process_data(combined_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Final memory: {final_memory:.1f} MB")
        print(f"   Memory increase: {memory_increase:.1f} MB")
        
        # Cleanup
        face_detector.cleanup()
        pose_estimator.cleanup()
        gesture_recognizer.cleanup()
        engagement_scorer.cleanup()
        
        gc.collect()
        
        # Memory should not increase excessively
        self.assertLess(memory_increase, 500)  # Less than 500MB increase

class TestFunctionality(unittest.TestCase):
    """Functionality tests for the engagement analyzer"""
    
    def test_face_detection_output_format(self):
        """Test face detection output format"""
        print("\n🧪 Testing Face Detection Output Format...")
        
        face_detector = FaceDetector(config.video.__dict__)
        face_detector.initialize()
        
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = face_detector.process_data(test_frame)
        
        # Check required fields
        self.assertIn('faces', result)
        self.assertIn('face_count', result)
        self.assertIn('attendance_count', result)
        self.assertIsInstance(result['faces'], list)
        self.assertIsInstance(result['face_count'], int)
        self.assertIsInstance(result['attendance_count'], int)
        
        face_detector.cleanup()
    
    def test_engagement_scoring_logic(self):
        """Test engagement scoring logic"""
        print("\n🧪 Testing Engagement Scoring Logic...")
        
        engagement_scorer = EngagementScorer(config.engagement.__dict__)
        engagement_scorer.initialize()
        
        # Test high engagement scenario
        high_engagement_data = {
            'face_detection': {'face_count': 10, 'attendance_count': 10},
            'pose_estimation': {'average_attention': 0.9, 'attention_distribution': {'high': 0.8, 'medium': 0.2, 'low': 0.0}},
            'gesture_recognition': {'participation_score': 0.8, 'participation_events': [1, 2, 3]},
            'audio_processing': {'engagement_metrics': {'audio_engagement_score': 0.9}}
        }
        
        result = engagement_scorer.process_data(high_engagement_data)
        
        self.assertIn('overall_engagement_score', result)
        self.assertIn('engagement_level', result)
        self.assertGreater(result['overall_engagement_score'], 0.7)
        self.assertEqual(result['engagement_level'], 'high')
        
        # Test low engagement scenario
        low_engagement_data = {
            'face_detection': {'face_count': 2, 'attendance_count': 10},
            'pose_estimation': {'average_attention': 0.2, 'attention_distribution': {'high': 0.1, 'medium': 0.2, 'low': 0.7}},
            'gesture_recognition': {'participation_score': 0.1, 'participation_events': []},
            'audio_processing': {'engagement_metrics': {'audio_engagement_score': 0.2}}
        }
        
        result = engagement_scorer.process_data(low_engagement_data)
        
        self.assertLess(result['overall_engagement_score'], 0.4)
        self.assertEqual(result['engagement_level'], 'low')
        
        engagement_scorer.cleanup()
    
    def test_communication_manager(self):
        """Test communication manager"""
        print("\n🧪 Testing Communication Manager...")
        
        # Mock the WebSocket and API clients
        with patch('src.utils.communication.WebSocketClient') as mock_ws, \
             patch('src.utils.communication.APIClient') as mock_api:
            
            mock_ws_instance = Mock()
            mock_ws_instance.connect.return_value = True
            mock_ws_instance.is_connected = True
            mock_ws.return_value = mock_ws_instance
            
            mock_api_instance = Mock()
            mock_api_instance.health_check.return_value = True
            mock_api.return_value = mock_api_instance
            
            comm_manager = CommunicationManager(config.communication.__dict__)
            
            # Test initialization
            self.assertTrue(comm_manager.start())
            
            # Test data sending
            test_data = {'test': 'data', 'timestamp': time.time()}
            comm_manager.send_engagement_data(test_data)
            
            # Verify WebSocket was called
            mock_ws_instance.connect.assert_called_once()
            
            comm_manager.stop()

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n🚀 Running Performance Benchmark...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add performance tests
    suite.addTest(TestPerformance('test_face_detection_performance'))
    suite.addTest(TestPerformance('test_pose_estimation_performance'))
    suite.addTest(TestPerformance('test_gesture_recognition_performance'))
    suite.addTest(TestPerformance('test_engagement_scoring_performance'))
    suite.addTest(TestPerformance('test_memory_usage'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All performance tests passed!")
        print("🎯 System is ready for 30fps real-time processing")
    else:
        print("❌ Some performance tests failed")
        print("⚠️  System may not meet real-time requirements")
    
    return result.wasSuccessful()

def run_functionality_tests():
    """Run functionality tests"""
    print("\n🧪 Running Functionality Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add functionality tests
    suite.addTest(TestFunctionality('test_face_detection_output_format'))
    suite.addTest(TestFunctionality('test_engagement_scoring_logic'))
    suite.addTest(TestFunctionality('test_communication_manager'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All functionality tests passed!")
    else:
        print("❌ Some functionality tests failed")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🎓 Real-time Classroom Engagement Analyzer - Test Suite")
    print("=" * 60)
    
    # Run all tests
    performance_ok = run_performance_benchmark()
    functionality_ok = run_functionality_tests()
    
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    if performance_ok and functionality_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready for deployment")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Please review and fix issues before deployment")
        exit(1)
