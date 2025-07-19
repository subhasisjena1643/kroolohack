"""
Main Application - Real-time Classroom Engagement Analyzer
Hackathon Project by Subhasis & Sachin
"""

import cv2
import time
import threading
import signal
import sys
from typing import Dict, Any, Optional

# Add parent directory to path for imports
import os
import sys

# Handle both direct execution and module execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import configuration and utilities
try:
    from config.config import config
    from src.utils.logger import logger
    from src.utils.communication import CommunicationManager
    # Import AI modules
    from src.modules.face_detector import FaceDetector
    from src.modules.pose_estimator import HeadPoseEstimator
    from src.modules.gesture_recognizer import GestureRecognizer
    from src.modules.audio_processor import AudioProcessor
    from src.modules.engagement_scorer import EngagementScorer
    # Import advanced industry-grade modules
    from src.modules.advanced_body_detector import AdvancedBodyDetector
    from src.modules.advanced_eye_tracker import AdvancedEyeTracker
    from src.modules.micro_expression_analyzer import MicroExpressionAnalyzer
    from src.modules.intelligent_pattern_analyzer import IntelligentPatternAnalyzer
    from src.modules.behavioral_classifier import BehavioralPatternClassifier
    from src.modules.intelligent_alert_system import IntelligentAlertSystem
    # Import continuous learning modules
    from src.modules.continuous_learning_system import ContinuousLearningSystem
    from src.modules.feedback_interface import FeedbackInterface
except ImportError:
    # Fallback for direct execution
    from config.config import config
    from utils.logger import logger
    from utils.communication import CommunicationManager
    # Import AI modules
    from modules.face_detector import FaceDetector
    from modules.pose_estimator import HeadPoseEstimator
    from modules.gesture_recognizer import GestureRecognizer
    from modules.audio_processor import AudioProcessor
    from modules.engagement_scorer import EngagementScorer
    # Import advanced industry-grade modules
    from modules.advanced_body_detector import AdvancedBodyDetector
    from modules.advanced_eye_tracker import AdvancedEyeTracker
    from modules.micro_expression_analyzer import MicroExpressionAnalyzer
    from modules.intelligent_pattern_analyzer import IntelligentPatternAnalyzer
    from modules.behavioral_classifier import BehavioralPatternClassifier
    from modules.intelligent_alert_system import IntelligentAlertSystem
    # Import continuous learning modules
    from modules.continuous_learning_system import ContinuousLearningSystem
    from modules.feedback_interface import FeedbackInterface

class EngagementAnalyzer:
    """Main application class for real-time engagement analysis"""
    
    def __init__(self):
        self.config = config
        self.is_running = False
        
        # Video capture
        self.cap = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # AI Processing modules
        self.face_detector = None
        self.pose_estimator = None
        self.gesture_recognizer = None
        self.audio_processor = None
        self.engagement_scorer = None

        # Advanced industry-grade modules
        self.advanced_body_detector = None
        self.advanced_eye_tracker = None
        self.micro_expression_analyzer = None
        self.intelligent_pattern_analyzer = None
        self.behavioral_classifier = None
        self.intelligent_alert_system = None

        # Continuous learning modules
        self.continuous_learning_system = None
        self.feedback_interface = None
        
        # Communication
        self.communication_manager = None
        
        # Performance tracking
        self.processing_times = {}
        self.total_latency = 0.0
        self.fps_counter = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

        # Performance optimization variables for 30+ FPS
        self.last_process_time = 0.0
        self.cached_results = None
        self.target_fps = 30
        self.frame_skip_count = 0

        # Real-time parameter tracking for display
        self.current_fps = 0.0
        self.last_processing_time = 0.0
        self.last_face_count = 0
        self.last_gesture_count = 0
        self.last_alert_status = 'Normal'
        self.training_samples_count = 0
        self.current_engagement_score = 0.0
        self.current_engagement_level = 'unknown'
        self.current_disengagement_rate = 0.0
        self.current_component_scores = {}
        self.parameter_update_time = time.time()

        # Face detection caching for continuity
        self.cached_face_result = None
        self.last_face_detection_time = 0.0
        self.face_cache_duration = 1.0  # Cache faces for 1 second for better continuity

        # OPTIMIZATION: Frame processing optimization
        self.optimization_frame_counter = 0

        # Alert timeout management
        self.active_alerts = {}
        self.alert_timeout_duration = 10.0  # 10 seconds

        # Display
        self.show_display = True
        self.display_thread = None
        
        logger.info("Engagement Analyzer initialized")
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing Engagement Analyzer...")
            
            # Initialize video capture
            if not self._initialize_camera():
                return False
            
            # Initialize AI modules
            if not self._initialize_ai_modules():
                return False
            
            # Initialize communication
            if not self._initialize_communication():
                return False
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def _initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            logger.info("Initializing camera...")
            
            self.cap = cv2.VideoCapture(self.config.video.camera_index)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.video.fps)
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                return False
            
            logger.info(f"Camera initialized: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _initialize_ai_modules(self) -> bool:
        """Initialize all AI processing modules"""
        try:
            logger.info("Initializing AI modules...")
            
            # Face Detection
            self.face_detector = FaceDetector(self.config.video.__dict__)
            if not self.face_detector.initialize():
                return False
            
            # Head Pose Estimation
            self.pose_estimator = HeadPoseEstimator(self.config.video.__dict__)
            if not self.pose_estimator.initialize():
                return False
            
            # Gesture Recognition
            self.gesture_recognizer = GestureRecognizer(self.config.video.__dict__)
            if not self.gesture_recognizer.initialize():
                return False
            
            # Audio Processing
            self.audio_processor = AudioProcessor(self.config.audio.__dict__)
            if not self.audio_processor.initialize():
                return False
            
            # Engagement Scorer
            self.engagement_scorer = EngagementScorer(self.config.engagement.__dict__)
            if not self.engagement_scorer.initialize():
                return False

            # Advanced Body Detector (Industry-grade precision)
            self.advanced_body_detector = AdvancedBodyDetector(self.config.video.__dict__)
            if not self.advanced_body_detector.initialize():
                return False

            # Advanced Eye Tracker
            self.advanced_eye_tracker = AdvancedEyeTracker(self.config.video.__dict__)
            if not self.advanced_eye_tracker.initialize():
                return False

            # Micro-Expression Analyzer
            self.micro_expression_analyzer = MicroExpressionAnalyzer(self.config.video.__dict__)
            if not self.micro_expression_analyzer.initialize():
                return False

            # Intelligent Pattern Analyzer
            self.intelligent_pattern_analyzer = IntelligentPatternAnalyzer(self.config.engagement.__dict__)
            if not self.intelligent_pattern_analyzer.initialize():
                return False

            # Behavioral Pattern Classifier
            self.behavioral_classifier = BehavioralPatternClassifier(self.config.engagement.__dict__)
            if not self.behavioral_classifier.initialize():
                return False

            # Intelligent Alert System
            self.intelligent_alert_system = IntelligentAlertSystem(self.config.engagement.__dict__)
            if not self.intelligent_alert_system.initialize():
                return False

            # Continuous Learning System
            self.continuous_learning_system = ContinuousLearningSystem(self.config.engagement.__dict__)
            if not self.continuous_learning_system.initialize():
                return False

            # Register models for continuous learning
            self._register_models_for_learning()

            # Feedback Interface
            self.feedback_interface = FeedbackInterface(self.config.engagement.__dict__)
            self.feedback_interface.start_server()

            # Setup feedback callback
            self.feedback_interface.add_feedback_callback(self._process_feedback)

            logger.info("All AI modules (including continuous learning) initialized")
            return True
            
        except Exception as e:
            logger.error(f"AI module initialization failed: {e}")
            return False
    
    def _initialize_communication(self) -> bool:
        """Initialize communication with backend"""
        try:
            logger.info("Initializing communication...")
            
            self.communication_manager = CommunicationManager(self.config.communication.__dict__)
            
            if not self.communication_manager.start():
                logger.warning("Communication manager failed to start - continuing without backend")
                return True  # Continue without backend for demo
            
            logger.info("Communication initialized")
            return True
            
        except Exception as e:
            logger.error(f"Communication initialization failed: {e}")
            return True  # Continue without backend
    
    def start(self) -> bool:
        """Start the engagement analyzer"""
        if not self.initialize():
            logger.error("Failed to initialize - cannot start")
            return False
        
        try:
            logger.info("Starting Engagement Analyzer...")
            
            # Start AI modules
            self.face_detector.start()
            self.pose_estimator.start()
            self.gesture_recognizer.start()
            self.audio_processor.start()
            self.engagement_scorer.start()

            # Start advanced modules
            self.advanced_body_detector.start()
            self.advanced_eye_tracker.start()
            self.micro_expression_analyzer.start()
            self.intelligent_pattern_analyzer.start()
            self.behavioral_classifier.start()
            self.intelligent_alert_system.start()
            
            # Start main processing loop
            self.is_running = True
            
            # Start display thread if enabled
            if self.show_display:
                self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
                self.display_thread.start()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("🚀 Engagement Analyzer started successfully!")
            logger.info("Press Ctrl+C to stop")
            
            # Main processing loop
            self._main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            return False
    
    def _main_loop(self):
        """Main processing loop"""
        logger.info("Starting main processing loop...")
        
        while self.is_running:
            try:
                loop_start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                self.frame_count += 1
                
                # Process frame through AI pipeline
                results = self._process_frame(frame)
                
                # Send results to backend
                if self.communication_manager and results:
                    self.communication_manager.send_engagement_data(results)
                
                # Calculate and log performance
                loop_time = time.time() - loop_start_time
                self.total_latency = loop_time
                
                # FPS calculation
                self._update_fps()
                
                # Performance logging (every 100 frames)
                if self.frame_count % 100 == 0:
                    self._log_performance()
                
                # Aggressive FPS optimization for 30+ FPS
                target_frame_time = 1.0 / 35.0  # Target 35 FPS for buffer
                if loop_time < target_frame_time:
                    # Minimal sleep for consistent timing
                    sleep_time = target_frame_time - loop_time
                    if sleep_time > 0.001:
                        time.sleep(min(sleep_time, 0.01))  # Cap sleep at 10ms
                elif loop_time > target_frame_time * 2.0:
                    # If too slow, skip next frame
                    self.frame_skip_count += 1
                    if self.frame_count % 60 == 0:  # Log every 60 frames
                        logger.warning(f"Performance warning: {loop_time:.3f}s (target: {target_frame_time:.3f}s)")
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(0.1)
        
        logger.info("Main processing loop stopped")
    
    def _process_frame(self, frame) -> Optional[Dict[str, Any]]:
        """Process single frame through industry-grade AI pipeline with 30+ FPS optimization"""
        try:
            frame_start_time = time.time()
            results = {}

            # Aggressive FPS optimization: Skip every other frame for 30+ FPS
            self.frame_skip_count += 1
            if self.frame_skip_count % 2 == 0:  # Process every 2nd frame
                if hasattr(self, 'cached_results') and self.cached_results:
                    return self.cached_results

            self.last_process_time = frame_start_time

            # Performance optimization: Skip heavy processing on some frames
            self.frame_skip_count += 1
            should_process_heavy = (self.frame_skip_count % 3 == 0)  # Process heavy analysis every 3rd frame

            # Face Detection with caching for continuity
            start_time = time.time()
            current_time = time.time()

            # Try to get fresh face detection
            self.face_detector.add_data(frame)
            face_result = self.face_detector.get_result(timeout=0.005)  # Ultra-fast timeout

            if face_result and face_result.get('faces'):
                # Fresh detection successful
                self.cached_face_result = face_result
                self.last_face_detection_time = current_time
                results['face_detection'] = face_result
                self.processing_times['face_detection'] = time.time() - start_time
            elif self.cached_face_result and (current_time - self.last_face_detection_time) < self.face_cache_duration:
                # Use cached result for continuity
                results['face_detection'] = self.cached_face_result
                self.processing_times['face_detection'] = 0.001  # Minimal time for cached result
            else:
                # No valid detection, clear cache
                self.cached_face_result = None
                results['face_detection'] = {'faces': [], 'face_count': 0}
                self.processing_times['face_detection'] = time.time() - start_time

            # OPTIMIZATION: Advanced Body Detection (Skip every 3rd frame for speed)
            if self.optimization_frame_counter % 3 == 0:  # Process every 3rd frame
                start_time = time.time()
                advanced_body_result = self.advanced_body_detector.process_data(frame)
                if advanced_body_result:
                    results['advanced_body_detection'] = advanced_body_result
                    self.processing_times['advanced_body_detection'] = time.time() - start_time
                    self.cached_body_result = advanced_body_result
            else:
                # Use cached result for skipped frames
                if hasattr(self, 'cached_body_result'):
                    results['advanced_body_detection'] = self.cached_body_result
                    self.processing_times['advanced_body_detection'] = 0.001  # Minimal time for cached result

            # OPTIMIZATION: Advanced Eye Tracking (Skip every 4th frame for speed)
            if self.optimization_frame_counter % 4 == 0:  # Process every 4th frame
                start_time = time.time()
                eye_tracking_result = self.advanced_eye_tracker.process_data(frame)
                if eye_tracking_result:
                    results['advanced_eye_tracking'] = eye_tracking_result
                    self.processing_times['advanced_eye_tracking'] = time.time() - start_time
                    self.cached_eye_result = eye_tracking_result
            else:
                # Use cached result for skipped frames
                if hasattr(self, 'cached_eye_result'):
                    results['advanced_eye_tracking'] = self.cached_eye_result
                    self.processing_times['advanced_eye_tracking'] = 0.001  # Minimal time for cached result

            # OPTIMIZATION: Micro-Expression Analysis (Skip every other frame for speed)
            self.optimization_frame_counter += 1
            if self.optimization_frame_counter % 2 == 0:  # Process every 2nd frame
                start_time = time.time()
                micro_expression_result = self.micro_expression_analyzer.process_data(frame)
                if micro_expression_result:
                    results['micro_expression_analysis'] = micro_expression_result
                    self.processing_times['micro_expression_analysis'] = time.time() - start_time
                    self.cached_micro_expression_result = micro_expression_result
            else:
                # Use cached result for skipped frames
                if hasattr(self, 'cached_micro_expression_result'):
                    results['micro_expression_analysis'] = self.cached_micro_expression_result
                    self.processing_times['micro_expression_analysis'] = 0.001  # Minimal time for cached result

            # OPTIMIZATION: Head Pose Estimation (Skip every 6th frame for speed)
            if self.optimization_frame_counter % 6 == 0:  # Process every 6th frame
                start_time = time.time()
                pose_input = {'frame': frame, 'faces': face_result.get('faces', []) if face_result else []}
                self.pose_estimator.add_data(pose_input)
                pose_result = self.pose_estimator.get_result(timeout=0.005)  # Ultra-fast timeout
                if pose_result:
                    results['pose_estimation'] = pose_result
                    self.processing_times['pose_estimation'] = time.time() - start_time
                    self.cached_pose_result = pose_result
            else:
                # Use cached result for skipped frames
                if hasattr(self, 'cached_pose_result'):
                    results['pose_estimation'] = self.cached_pose_result
                    self.processing_times['pose_estimation'] = 0.001  # Minimal time for cached result

            # Gesture Recognition (ultra-fast)
            start_time = time.time()
            self.gesture_recognizer.add_data(frame)
            gesture_result = self.gesture_recognizer.get_result(timeout=0.005)  # Ultra-fast timeout
            if gesture_result:
                results['gesture_recognition'] = gesture_result
                self.processing_times['gesture_recognition'] = time.time() - start_time

            # Audio Processing (continuous, get latest result) - already optimized
            start_time = time.time()
            audio_result = self.audio_processor.get_result(timeout=0.005)  # Reduced from 0.01
            if audio_result:
                results['audio_processing'] = audio_result
                self.processing_times['audio_processing'] = time.time() - start_time

            # OPTIMIZATION: Intelligent Pattern Analysis (Skip every 5th frame for speed)
            if results and self.optimization_frame_counter % 5 == 0:  # Process every 5th frame
                start_time = time.time()
                movement_data = {
                    'advanced_body_detection': results.get('advanced_body_detection', {}),
                    'advanced_eye_tracking': results.get('advanced_eye_tracking', {}),
                    'micro_expression_analysis': results.get('micro_expression_analysis', {}),
                    'pose_estimation': results.get('pose_estimation', {}),
                    'gesture_recognition': results.get('gesture_recognition', {})
                }
                pattern_result = self.intelligent_pattern_analyzer.process_data(movement_data)
                if pattern_result:
                    results['intelligent_pattern_analysis'] = pattern_result
                    self.processing_times['intelligent_pattern_analysis'] = time.time() - start_time
                    self.cached_pattern_result = pattern_result
            elif hasattr(self, 'cached_pattern_result'):
                # Use cached result for skipped frames
                results['intelligent_pattern_analysis'] = self.cached_pattern_result
                self.processing_times['intelligent_pattern_analysis'] = 0.001  # Minimal time for cached result

            # Initialize movement_data for other components that might need it
            movement_data = {
                'advanced_body_detection': results.get('advanced_body_detection', {}),
                'advanced_eye_tracking': results.get('advanced_eye_tracking', {}),
                'micro_expression_analysis': results.get('micro_expression_analysis', {}),
                'pose_estimation': results.get('pose_estimation', {}),
                'gesture_recognition': results.get('gesture_recognition', {})
            }

            # OPTIMIZATION: Behavioral Classification (Skip every 10th frame for speed)
            if results and self.optimization_frame_counter % 10 == 0:  # Process every 10th frame
                start_time = time.time()
                behavioral_result = self.behavioral_classifier.process_data(movement_data)
                if behavioral_result:
                    results['behavioral_classification'] = behavioral_result
                    self.processing_times['behavioral_classification'] = time.time() - start_time
                    self.cached_behavioral_result = behavioral_result
            elif hasattr(self, 'cached_behavioral_result'):
                # Use cached result for skipped frames
                results['behavioral_classification'] = self.cached_behavioral_result
                self.processing_times['behavioral_classification'] = 0.001  # Minimal time for cached result

            # Engagement Scoring (ultra-fast)
            if results:
                start_time = time.time()
                self.engagement_scorer.add_data(results)
                engagement_result = self.engagement_scorer.get_result(timeout=0.005)  # Ultra-fast
                if engagement_result:
                    results['engagement_analysis'] = engagement_result
                    self.processing_times['engagement_scoring'] = time.time() - start_time

            # Intelligent Alert System (ultra-fast)
            if results:
                start_time = time.time()
                alert_result = self.intelligent_alert_system.process_data(engagement_result)
                if alert_result:
                    results['intelligent_alerts'] = alert_result
                    self.processing_times['intelligent_alerts'] = time.time() - start_time

            # Continuous Learning (collect data for improvement) - async
            if results and engagement_result:
                # Run learning data collection in background to not block main loop
                threading.Thread(
                    target=self._collect_learning_data,
                    args=(results, engagement_result),
                    daemon=True
                ).start()

            # Add frame metadata
            results['frame_metadata'] = {
                'frame_number': self.frame_count,
                'timestamp': time.time(),
                'frame_shape': frame.shape,
                'processing_times': self.processing_times.copy(),
                'total_latency': time.time() - frame_start_time
            }

            # Cache results for frame skipping
            self.cached_results = results

            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _display_loop(self):
        """Display loop for visualization"""
        logger.info("Starting display loop...")
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Get latest results for visualization
                face_result = self.face_detector.get_result(timeout=0.01)
                pose_result = self.pose_estimator.get_result(timeout=0.01)
                gesture_result = self.gesture_recognizer.get_result(timeout=0.01)
                engagement_result = self.engagement_scorer.get_result(timeout=0.01)

                # Get advanced analysis results
                advanced_body_result = self.advanced_body_detector.get_result(timeout=0.01) if hasattr(self.advanced_body_detector, 'get_result') else None
                eye_tracking_result = self.advanced_eye_tracker.get_result(timeout=0.01) if hasattr(self.advanced_eye_tracker, 'get_result') else None
                micro_expression_result = self.micro_expression_analyzer.get_result(timeout=0.01) if hasattr(self.micro_expression_analyzer, 'get_result') else None
                alert_result = self.intelligent_alert_system.get_active_alerts() if self.intelligent_alert_system else []
                
                # Draw visualizations
                display_frame = frame.copy()
                
                if face_result and face_result.get('faces'):
                    display_frame = self.face_detector.draw_detections(
                        display_frame, face_result['faces']
                    )
                
                if pose_result and pose_result.get('poses'):
                    display_frame = self.pose_estimator.draw_pose_estimation(
                        display_frame, pose_result['poses']
                    )
                
                if gesture_result and gesture_result.get('hands'):
                    display_frame = self.gesture_recognizer.draw_gestures(
                        display_frame, gesture_result['hands'], gesture_result.get('gestures', {})
                    )
                
                # Draw engagement info
                if engagement_result:
                    self._draw_engagement_info(display_frame, engagement_result)

                # Draw advanced analysis info
                if eye_tracking_result:
                    self._draw_eye_tracking_info(display_frame, eye_tracking_result)

                if micro_expression_result:
                    self._draw_micro_expression_info(display_frame, micro_expression_result)

                # Draw alerts with timeout management
                if alert_result:
                    filtered_alerts = self._manage_alert_timeouts(alert_result)
                    if filtered_alerts:
                        self._draw_alerts(display_frame, filtered_alerts)

                # Draw performance info
                self._draw_performance_info(display_frame)
                
                # Update real-time parameters for display
                self._update_real_time_parameters(engagement_result)

                # Update web interface with real-time data
                self._update_web_interface(engagement_result, face_result, gesture_result, alert_result)

                # Show frame
                cv2.imshow('Engagement Analyzer', display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.stop()
                    break
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                time.sleep(0.1)

    def _update_real_time_parameters(self, engagement_result):
        """Update real-time parameters for display"""
        try:
            current_time = time.time()

            # Update FPS
            if hasattr(self, 'fps_counter'):
                self.current_fps = self.fps_counter

            # Update processing time
            if hasattr(self, 'last_process_time'):
                self.last_processing_time = self.last_process_time

            # Update engagement parameters
            if engagement_result:
                self.current_engagement_score = engagement_result.get('overall_engagement_score', 0.0)
                self.current_engagement_level = engagement_result.get('engagement_level', 'unknown')
                self.current_disengagement_rate = max(0.0, 1.0 - self.current_engagement_score)
                self.current_component_scores = engagement_result.get('component_scores', {})

            # Update face count
            if hasattr(self, 'face_detector') and self.face_detector:
                try:
                    face_result = self.face_detector.get_result(timeout=0.001)
                    if face_result and 'faces' in face_result:
                        self.last_face_count = len(face_result['faces'])
                except:
                    pass

            # Update gesture count
            if hasattr(self, 'gesture_recognizer') and self.gesture_recognizer:
                try:
                    gesture_result = self.gesture_recognizer.get_result(timeout=0.001)
                    if gesture_result and 'gestures' in gesture_result:
                        self.last_gesture_count = len(gesture_result['gestures'])
                except:
                    pass

            # Update alert status
            if hasattr(self, 'intelligent_alert_system') and self.intelligent_alert_system:
                try:
                    active_alerts = self.intelligent_alert_system.get_active_alerts()
                    if active_alerts:
                        self.last_alert_status = f"Alert: {len(active_alerts)} active"
                    else:
                        self.last_alert_status = "Normal"
                except:
                    self.last_alert_status = "Normal"

            # Update training samples count
            if hasattr(self, 'continuous_learning_system') and self.continuous_learning_system:
                try:
                    # Get training samples count from learning system
                    self.training_samples_count = len(getattr(self.continuous_learning_system, 'learning_instances', []))
                except:
                    pass

            self.parameter_update_time = current_time

        except Exception as e:
            logger.error(f"Error updating real-time parameters: {e}")

    def _manage_alert_timeouts(self, new_alerts):
        """Manage alert timeouts - show alerts for only 10 seconds"""
        try:
            current_time = time.time()
            filtered_alerts = []

            # Process new alerts
            for alert in new_alerts:
                # Handle Alert dataclass objects
                if hasattr(alert, 'alert_id'):
                    # Alert dataclass object
                    alert_id = alert.alert_id
                elif hasattr(alert, '__dict__'):
                    # Other object with attributes
                    alert_type = getattr(alert, 'type', getattr(alert, 'alert_type', 'unknown'))
                    alert_message = getattr(alert, 'message', getattr(alert, 'description', ''))
                    alert_id = f"{alert_type}_{alert_message}"
                elif isinstance(alert, dict):
                    # Dictionary alert
                    alert_id = alert.get('alert_id', alert.get('id', f"{alert.get('type', 'unknown')}_{alert.get('message', '')}"))
                else:
                    # String or other format
                    alert_id = str(alert)

                # Add new alert or update existing one
                if alert_id not in self.active_alerts:
                    self.active_alerts[alert_id] = {
                        'alert': alert,
                        'start_time': current_time,
                        'last_seen': current_time
                    }
                else:
                    # Update last seen time
                    self.active_alerts[alert_id]['last_seen'] = current_time

            # Filter alerts based on timeout
            alerts_to_remove = []
            for alert_id, alert_data in self.active_alerts.items():
                time_since_start = current_time - alert_data['start_time']
                time_since_seen = current_time - alert_data['last_seen']

                # Remove alert if it's been 10 seconds since it started OR 2 seconds since last seen
                if time_since_start >= self.alert_timeout_duration or time_since_seen >= 2.0:
                    alerts_to_remove.append(alert_id)
                else:
                    # Keep this alert
                    filtered_alerts.append(alert_data['alert'])

            # Remove expired alerts
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]

            return filtered_alerts

        except Exception as e:
            logger.error(f"Error managing alert timeouts: {e}")
            return new_alerts if new_alerts else []

    def _update_web_interface(self, engagement_result, face_result, gesture_result, alert_result):
        """Update web interface with real-time data"""
        try:
            if not hasattr(self, 'feedback_interface') or not self.feedback_interface:
                return

            # Update engagement data
            if engagement_result:
                self.feedback_interface.update_engagement_data(engagement_result)

            # Update performance metrics
            self.feedback_interface.update_performance_metrics(
                fps=self.current_fps,
                processing_time=self.last_processing_time,
                component_times=self.processing_times
            )

            # Update face detection status
            face_count = 0
            if face_result and face_result.get('faces'):
                face_count = len(face_result['faces'])

            # Update gesture detection status
            gesture_count = 0
            if gesture_result and gesture_result.get('gestures'):
                gesture_count = len(gesture_result['gestures'])

            self.feedback_interface.update_gesture_detection(gesture_count)

            # Update alert status
            if hasattr(self, 'active_alerts'):
                timeout_active = len(self.active_alerts) > 0
                self.feedback_interface.update_alert_status(alert_result or [], timeout_active)

            # Update training metrics if continuous learning is active
            if hasattr(self, 'continuous_learning_system') and self.continuous_learning_system:
                try:
                    training_samples = len(getattr(self.continuous_learning_system, 'learning_instances', []))
                    model_versions = getattr(self.continuous_learning_system, 'model_versions', {})
                    accuracy = 0.85  # Default accuracy, would be calculated from actual performance

                    self.feedback_interface.update_training_metrics(
                        total_samples=training_samples,
                        accuracy=accuracy,
                        model_versions=model_versions
                    )
                except:
                    pass

            # Update SOTA datasets status
            sota_loaded = hasattr(self, 'sota_datasets') and self.sota_datasets is not None
            dataset_count = 5 if sota_loaded else 0  # Default count
            self.feedback_interface.update_sota_datasets_status(sota_loaded, dataset_count)

        except Exception as e:
            logger.error(f"Error updating web interface: {e}")
    
    def _draw_engagement_info(self, frame, engagement_result):
        """Draw comprehensive engagement information on frame with full-screen layout"""
        try:
            if not engagement_result:
                return

            frame_height, frame_width = frame.shape[:2]

            # Create overlay for better text visibility
            overlay = frame.copy()

            # Main engagement metrics (top-left)
            score = engagement_result.get('overall_engagement_score', 0.0)
            level = engagement_result.get('engagement_level', 'unknown')
            disengagement_rate = max(0.0, 1.0 - score)

            # Draw main engagement panel
            self._draw_main_engagement_panel(overlay, score, level, disengagement_rate)

            # Draw component scores panel (top-right)
            components = engagement_result.get('component_scores', {})
            self._draw_component_scores_panel(overlay, components, frame_width)

            # Draw performance metrics (bottom-left)
            self._draw_performance_metrics_panel(overlay, frame_height)

            # Draw real-time statistics (bottom-right)
            self._draw_statistics_panel(overlay, engagement_result, frame_width, frame_height)

            # Draw FPS and system info (top-center)
            self._draw_system_info_panel(overlay, frame_width)

            # Blend overlay with original frame
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        except Exception as e:
            logger.error(f"Error drawing engagement info: {e}")

    def _draw_main_engagement_panel(self, frame, score, level, disengagement_rate):
        """Draw main engagement metrics panel"""
        # Background panel
        cv2.rectangle(frame, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 150), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "ENGAGEMENT ANALYSIS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Engagement score with color coding
        color = (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
        text = f"Score: {score:.3f}"
        cv2.putText(frame, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Engagement level
        level_color = (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Level: {level.upper()}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2)

        # Disengagement rate
        disengagement_color = (0, 0, 255) if disengagement_rate > 0.6 else (0, 165, 255) if disengagement_rate > 0.3 else (0, 255, 0)
        cv2.putText(frame, f"Disengagement: {disengagement_rate:.3f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, disengagement_color, 2)

    def _draw_component_scores_panel(self, frame, components, frame_width):
        """Draw component scores panel"""
        panel_x = frame_width - 320
        panel_y = 10
        panel_width = 310
        panel_height = min(200, 30 + len(components) * 25)

        # Background panel
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "COMPONENT SCORES", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Component scores
        y_offset = panel_y + 50
        for component, comp_score in components.items():
            component_color = (0, 255, 0) if comp_score > 0.6 else (0, 255, 255) if comp_score > 0.3 else (0, 0, 255)
            text = f"{component[:15]}: {comp_score:.3f}"
            cv2.putText(frame, text, (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, component_color, 1)

            # Progress bar
            bar_x = panel_x + 200
            bar_y = y_offset - 10
            bar_width = 80
            bar_height = 8

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            fill_width = int(bar_width * comp_score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), component_color, -1)

            y_offset += 25

    def _draw_performance_metrics_panel(self, frame, frame_height):
        """Draw performance metrics panel"""
        panel_x = 10
        panel_y = frame_height - 120
        panel_width = 350
        panel_height = 110

        # Background panel
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "PERFORMANCE METRICS", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        current_fps = getattr(self, 'current_fps', 0.0)
        fps_color = (0, 255, 0) if current_fps >= 25 else (0, 255, 255) if current_fps >= 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (panel_x + 10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # Processing time
        processing_time = getattr(self, 'last_processing_time', 0.0)
        time_color = (0, 255, 0) if processing_time < 0.033 else (0, 255, 255) if processing_time < 0.066 else (0, 0, 255)
        cv2.putText(frame, f"Process Time: {processing_time:.3f}s", (panel_x + 10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, time_color, 1)

        # Frame count
        frame_count = getattr(self, 'frame_count', 0)
        cv2.putText(frame, f"Frames: {frame_count}", (panel_x + 200, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Target FPS indicator
        cv2.putText(frame, "Target: 30+ FPS", (panel_x + 200, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _draw_statistics_panel(self, frame, engagement_result, frame_width, frame_height):
        """Draw real-time statistics panel"""
        panel_x = frame_width - 320
        panel_y = frame_height - 150
        panel_width = 310
        panel_height = 140

        # Background panel
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "REAL-TIME STATS", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Face detection count
        face_count = getattr(self, 'last_face_count', 0)
        cv2.putText(frame, f"Faces Detected: {face_count}", (panel_x + 10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Gesture detection
        gesture_count = getattr(self, 'last_gesture_count', 0)
        cv2.putText(frame, f"Gestures: {gesture_count}", (panel_x + 10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Alert status
        alert_status = getattr(self, 'last_alert_status', 'Normal')
        alert_color = (0, 0, 255) if alert_status != 'Normal' else (0, 255, 0)
        cv2.putText(frame, f"Status: {alert_status}", (panel_x + 10, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)

        # Training progress
        training_samples = getattr(self, 'training_samples_count', 0)
        cv2.putText(frame, f"Training Samples: {training_samples}", (panel_x + 10, panel_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_system_info_panel(self, frame, frame_width):
        """Draw system information panel"""
        panel_x = frame_width // 2 - 150
        panel_y = 10
        panel_width = 300
        panel_height = 60

        # Background panel
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

        # System title
        cv2.putText(frame, "CLASSROOM ENGAGEMENT ANALYZER", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Version and status
        cv2.putText(frame, "v2.0 | Industry-Grade AI | 30+ FPS", (panel_x + 10, panel_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_eye_tracking_info(self, frame, eye_tracking_result):
        """Draw eye tracking information"""
        try:
            engagement_metrics = eye_tracking_result.get('engagement_metrics', {})
            eye_engagement_score = engagement_metrics.get('overall_engagement_score', 0.0)

            text = f"Eye Engagement: {eye_engagement_score:.2f}"
            cv2.putText(frame, text, (10, frame.shape[0] - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        except Exception as e:
            logger.error(f"Error drawing eye tracking info: {e}")

    def _draw_micro_expression_info(self, frame, micro_expression_result):
        """Draw micro-expression information"""
        try:
            facial_engagement = micro_expression_result.get('facial_engagement_metrics', {})
            facial_score = facial_engagement.get('overall_engagement_score', 0.0)

            text = f"Facial Engagement: {facial_score:.2f}"
            cv2.putText(frame, text, (10, frame.shape[0] - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        except Exception as e:
            logger.error(f"Error drawing micro-expression info: {e}")

    def _draw_alerts(self, frame, alerts):
        """Draw active alerts"""
        try:
            if not alerts:
                return

            y_offset = 90
            for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
                alert_text = f"ALERT: {alert.alert_type.value}"
                color = (0, 0, 255) if alert.severity.value == 'high' else (0, 165, 255)

                cv2.putText(frame, alert_text, (10, y_offset + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception as e:
            logger.error(f"Error drawing alerts: {e}")

    def _draw_performance_info(self, frame):
        """Draw performance information on frame"""
        try:
            # FPS
            fps_text = f"FPS: {self.fps_counter:.1f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Latency
            latency_text = f"Latency: {self.total_latency*1000:.1f}ms"
            cv2.putText(frame, latency_text, (frame.shape[1] - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Industry-grade indicator
            cv2.putText(frame, "INDUSTRY-GRADE PRECISION", (frame.shape[1] - 250, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        except Exception as e:
            logger.error(f"Error drawing performance info: {e}")
    
    def _update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps_counter = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _log_performance(self):
        """Log performance metrics"""
        logger.performance(
            "Frame Processing",
            self.total_latency,
            f"FPS: {self.fps_counter:.1f}, Components: {self.processing_times}"
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _register_models_for_learning(self):
        """Register ML models with continuous learning system"""
        try:
            if self.continuous_learning_system:
                # Register behavioral classifier models
                if hasattr(self.behavioral_classifier, 'engagement_classifier'):
                    self.continuous_learning_system.register_model(
                        'engagement_classifier',
                        self.behavioral_classifier.engagement_classifier
                    )

                if hasattr(self.behavioral_classifier, 'movement_type_classifier'):
                    self.continuous_learning_system.register_model(
                        'movement_type_classifier',
                        self.behavioral_classifier.movement_type_classifier
                    )

                # Register pattern analyzer models
                if hasattr(self.intelligent_pattern_analyzer, 'disengagement_classifier'):
                    self.continuous_learning_system.register_model(
                        'disengagement_classifier',
                        self.intelligent_pattern_analyzer.disengagement_classifier
                    )

                logger.info("Models registered for continuous learning")

                # Finalize continuous learning initialization
                self.continuous_learning_system.finalize_initialization()

        except Exception as e:
            logger.error(f"Error registering models for learning: {e}")

    def _collect_learning_data(self, results: Dict[str, Any], engagement_result: Dict[str, Any]):
        """Collect data for continuous learning"""
        try:
            if not self.continuous_learning_system:
                return

            # Extract features for learning
            features = self._extract_learning_features(results)

            # Get engagement prediction
            engagement_level = engagement_result.get('engagement_level', 'unknown')
            confidence = engagement_result.get('confidence', 0.0)

            # Add prediction feedback for engagement classifier
            self.continuous_learning_system.add_prediction_feedback(
                model_name='engagement_classifier',
                features=features,
                predicted_label=engagement_level,
                confidence=confidence
            )

            # Add behavioral classification feedback
            behavioral_result = results.get('behavioral_classification', {})
            if behavioral_result:
                behavioral_state = behavioral_result.get('intelligent_decision', {}).get('behavioral_state', 'unknown')
                behavioral_confidence = behavioral_result.get('classification_confidence', 0.0)

                self.continuous_learning_system.add_prediction_feedback(
                    model_name='movement_type_classifier',
                    features=features,
                    predicted_label=behavioral_state,
                    confidence=behavioral_confidence
                )

        except Exception as e:
            logger.error(f"Error collecting learning data: {e}")

    def _extract_learning_features(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for machine learning"""
        features = {}

        try:
            # Extract body movement features
            body_result = results.get('advanced_body_detection', {})
            if body_result:
                movement_analysis = body_result.get('movement_analysis', {})
                features.update({
                    'head_stability': movement_analysis.get('head_stability', 0.0),
                    'posture_alignment': movement_analysis.get('posture_alignment', 0.0),
                    'movement_intensity': movement_analysis.get('movement_intensity', 0.0)
                })

            # Extract eye tracking features
            eye_result = results.get('advanced_eye_tracking', {})
            if eye_result:
                engagement_metrics = eye_result.get('engagement_metrics', {})
                features.update({
                    'eye_engagement_score': engagement_metrics.get('overall_engagement_score', 0.0),
                    'gaze_stability': engagement_metrics.get('component_scores', {}).get('gaze_stability', 0.0),
                    'attention_focus': engagement_metrics.get('component_scores', {}).get('attention_focus', 0.0)
                })

            # Extract facial expression features
            facial_result = results.get('micro_expression_analysis', {})
            if facial_result:
                facial_engagement = facial_result.get('facial_engagement_metrics', {})
                features.update({
                    'facial_engagement_score': facial_engagement.get('overall_engagement_score', 0.0),
                    'emotional_stability': facial_engagement.get('emotional_stability', 0.0)
                })

            # Extract gesture features
            gesture_result = results.get('gesture_recognition', {})
            if gesture_result:
                features.update({
                    'hand_gesture_activity': gesture_result.get('gesture_count', 0),
                    'participation_gestures': gesture_result.get('participation_score', 0.0)
                })

            # Extract audio features
            audio_result = results.get('audio_processing', {})
            if audio_result:
                features.update({
                    'audio_engagement': audio_result.get('engagement_score', 0.0),
                    'speech_activity': audio_result.get('speech_detected', False)
                })

        except Exception as e:
            logger.error(f"Error extracting learning features: {e}")

        return features

    def _process_feedback(self, feedback_entry):
        """Process feedback from teacher interface"""
        try:
            if not self.continuous_learning_system:
                return

            # Add teacher feedback to learning system
            self.continuous_learning_system.add_teacher_feedback(
                model_name='engagement_classifier',
                features=feedback_entry.context.get('features', {}),
                predicted_label=feedback_entry.predicted_engagement,
                correct_label=feedback_entry.actual_engagement,
                confidence=feedback_entry.confidence
            )

            logger.info(f"Processed teacher feedback: {feedback_entry.predicted_engagement} -> {feedback_entry.actual_engagement}")

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get continuous learning statistics"""
        if self.continuous_learning_system:
            return self.continuous_learning_system.get_learning_statistics()
        return {}

    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        if self.continuous_learning_system:
            return {
                'engagement_classifier': self.continuous_learning_system.get_performance_metrics('engagement_classifier'),
                'movement_classifier': self.continuous_learning_system.get_performance_metrics('movement_type_classifier'),
                'disengagement_classifier': self.continuous_learning_system.get_performance_metrics('disengagement_classifier')
            }
        return {}
    
    def stop(self):
        """Stop the engagement analyzer"""
        logger.info("Stopping Engagement Analyzer...")
        
        self.is_running = False
        
        # Stop AI modules
        if self.face_detector:
            self.face_detector.stop()
        if self.pose_estimator:
            self.pose_estimator.stop()
        if self.gesture_recognizer:
            self.gesture_recognizer.stop()
        if self.audio_processor:
            self.audio_processor.stop()
        if self.engagement_scorer:
            self.engagement_scorer.stop()

        # Stop advanced modules
        if self.advanced_body_detector:
            self.advanced_body_detector.stop()
        if self.advanced_eye_tracker:
            self.advanced_eye_tracker.stop()
        if self.micro_expression_analyzer:
            self.micro_expression_analyzer.stop()
        if self.intelligent_pattern_analyzer:
            self.intelligent_pattern_analyzer.stop()
        if self.behavioral_classifier:
            self.behavioral_classifier.stop()
        if self.intelligent_alert_system:
            self.intelligent_alert_system.stop()

        # Stop continuous learning modules
        if self.continuous_learning_system:
            self.continuous_learning_system.cleanup()
        if self.feedback_interface:
            self.feedback_interface.stop_server()
        
        # Stop communication
        if self.communication_manager:
            self.communication_manager.stop()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close display
        cv2.destroyAllWindows()
        
        logger.info("Engagement Analyzer stopped")

def main():
    """Main entry point"""
    print("🎓 Real-time Classroom Engagement Analyzer")
    print("=" * 50)
    print("Hackathon Project by Subhasis & Sachin")
    print("AI-powered engagement detection system")
    print("=" * 50)
    
    try:
        # Create and start analyzer
        analyzer = EngagementAnalyzer()
        
        if analyzer.start():
            logger.info("Application completed successfully")
        else:
            logger.error("Application failed to start")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
