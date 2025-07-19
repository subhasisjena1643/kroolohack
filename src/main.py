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

# Import configuration and utilities
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
                
                # Maintain target FPS
                target_frame_time = 1.0 / self.config.video.fps
                if loop_time < target_frame_time:
                    time.sleep(target_frame_time - loop_time)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(0.1)
        
        logger.info("Main processing loop stopped")
    
    def _process_frame(self, frame) -> Optional[Dict[str, Any]]:
        """Process single frame through industry-grade AI pipeline"""
        try:
            results = {}

            # Face Detection
            start_time = time.time()
            self.face_detector.add_data(frame)
            face_result = self.face_detector.get_result(timeout=0.05)
            if face_result:
                results['face_detection'] = face_result
                self.processing_times['face_detection'] = time.time() - start_time

            # Advanced Body Detection (Industry-grade precision)
            start_time = time.time()
            advanced_body_result = self.advanced_body_detector.process_data(frame)
            if advanced_body_result:
                results['advanced_body_detection'] = advanced_body_result
                self.processing_times['advanced_body_detection'] = time.time() - start_time

            # Advanced Eye Tracking
            start_time = time.time()
            eye_tracking_result = self.advanced_eye_tracker.process_data(frame)
            if eye_tracking_result:
                results['advanced_eye_tracking'] = eye_tracking_result
                self.processing_times['advanced_eye_tracking'] = time.time() - start_time

            # Micro-Expression Analysis
            start_time = time.time()
            micro_expression_result = self.micro_expression_analyzer.process_data(frame)
            if micro_expression_result:
                results['micro_expression_analysis'] = micro_expression_result
                self.processing_times['micro_expression_analysis'] = time.time() - start_time

            # Head Pose Estimation (needs frame and face data)
            start_time = time.time()
            pose_input = {'frame': frame, 'faces': face_result.get('faces', []) if face_result else []}
            self.pose_estimator.add_data(pose_input)
            pose_result = self.pose_estimator.get_result(timeout=0.05)
            if pose_result:
                results['pose_estimation'] = pose_result
                self.processing_times['pose_estimation'] = time.time() - start_time

            # Gesture Recognition
            start_time = time.time()
            self.gesture_recognizer.add_data(frame)
            gesture_result = self.gesture_recognizer.get_result(timeout=0.05)
            if gesture_result:
                results['gesture_recognition'] = gesture_result
                self.processing_times['gesture_recognition'] = time.time() - start_time

            # Audio Processing (continuous, get latest result)
            start_time = time.time()
            audio_result = self.audio_processor.get_result(timeout=0.01)
            if audio_result:
                results['audio_processing'] = audio_result
                self.processing_times['audio_processing'] = time.time() - start_time

            # Intelligent Pattern Analysis (combine movement data)
            if results:
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

            # Behavioral Classification
            if results:
                start_time = time.time()
                behavioral_result = self.behavioral_classifier.process_data(movement_data)
                if behavioral_result:
                    results['behavioral_classification'] = behavioral_result
                    self.processing_times['behavioral_classification'] = time.time() - start_time

            # Engagement Scoring (combine all results)
            if results:
                start_time = time.time()
                self.engagement_scorer.add_data(results)
                engagement_result = self.engagement_scorer.get_result(timeout=0.05)
                if engagement_result:
                    results['engagement_analysis'] = engagement_result
                    self.processing_times['engagement_scoring'] = time.time() - start_time

            # Intelligent Alert System (final analysis)
            if results:
                start_time = time.time()
                alert_result = self.intelligent_alert_system.process_data(engagement_result)
                if alert_result:
                    results['intelligent_alerts'] = alert_result
                    self.processing_times['intelligent_alerts'] = time.time() - start_time

            # Continuous Learning (collect data for improvement)
            if results and engagement_result:
                self._collect_learning_data(results, engagement_result)
            
            # Add frame metadata
            results['frame_metadata'] = {
                'frame_number': self.frame_count,
                'timestamp': time.time(),
                'frame_shape': frame.shape,
                'processing_times': self.processing_times.copy(),
                'total_latency': self.total_latency
            }
            
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

                # Draw alerts
                if alert_result:
                    self._draw_alerts(display_frame, alert_result)

                # Draw performance info
                self._draw_performance_info(display_frame)
                
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
    
    def _draw_engagement_info(self, frame, engagement_result):
        """Draw engagement information on frame"""
        try:
            score = engagement_result.get('overall_engagement_score', 0.0)
            level = engagement_result.get('engagement_level', 'unknown')
            
            # Draw engagement score
            text = f"Engagement: {score:.2f} ({level})"
            cv2.putText(frame, text, (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw component scores
            components = engagement_result.get('component_scores', {})
            y_offset = frame.shape[0] - 40
            for component, score in components.items():
                text = f"{component}: {score:.2f}"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
                
        except Exception as e:
            logger.error(f"Error drawing engagement info: {e}")
    
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
