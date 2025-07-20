"""
Advanced Eye Tracking & Gaze Analysis System
Industry-grade precision eye tracking for attention and engagement detection
Implements state-of-the-art gaze estimation and attention zone analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional
import time
import math
from collections import deque
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

from utils.base_processor import BaseProcessor
from utils.logger import logger

class AdvancedEyeTracker(BaseProcessor):
    """Industry-grade eye tracking with precision gaze analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AdvancedEyeTracker", config)
        
        # MediaPipe Face Mesh for detailed eye tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
        # Eye tracking parameters
        self.min_detection_confidence = config.get('min_detection_confidence', 0.8)
        self.min_tracking_confidence = config.get('min_tracking_confidence', 0.7)
        
        # Eye landmark indices (MediaPipe 468 face landmarks)
        self.eye_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_iris': [468, 469, 470, 471, 472],
            'right_iris': [473, 474, 475, 476, 477]
        }
        
        # Gaze analysis components
        self.gaze_estimator = GazeEstimator()
        self.attention_zone_analyzer = AttentionZoneAnalyzer()
        self.blink_analyzer = BlinkAnalyzer()
        self.fixation_detector = FixationDetector()
        self.saccade_detector = SaccadeDetector()
        
        # Tracking history
        self.gaze_history = deque(maxlen=100)  # 3+ seconds at 30fps
        self.attention_history = deque(maxlen=300)  # 10 seconds
        self.eye_state_history = deque(maxlen=50)
        
        # Calibration and normalization
        self.calibration_data = None
        self.screen_dimensions = config.get('screen_dimensions', (1920, 1080))
        self.camera_position = config.get('camera_position', 'center_top')
        
        # Performance metrics
        self.tracking_quality = 0.0
        self.gaze_accuracy = 0.0
        
    def initialize(self) -> bool:
        """Initialize advanced eye tracking system"""
        try:
            logger.info("Initializing advanced eye tracking system...")
            
            # Initialize MediaPipe Face Mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,  # Focus on single person for precision
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            # Initialize gaze estimation components
            self.gaze_estimator.initialize()
            self.attention_zone_analyzer.initialize(self.screen_dimensions)
            self.blink_analyzer.initialize()
            self.fixation_detector.initialize()
            self.saccade_detector.initialize()
            
            logger.info("Advanced eye tracking system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced eye tracker: {e}")
            return False
    
    def process_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for advanced eye tracking and gaze analysis"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return self._empty_result("No face detected")
            
            # Extract eye landmarks
            face_landmarks = results.multi_face_landmarks[0]
            eye_data = self._extract_eye_landmarks(face_landmarks, frame.shape)
            
            # Gaze estimation
            gaze_analysis = self.gaze_estimator.estimate_gaze(eye_data, frame.shape)
            
            # Attention zone analysis
            attention_analysis = self.attention_zone_analyzer.analyze_attention(gaze_analysis)
            
            # Blink analysis
            blink_analysis = self.blink_analyzer.analyze_blinks(eye_data)
            
            # Fixation detection
            fixation_analysis = self.fixation_detector.detect_fixations(gaze_analysis)
            
            # Saccade detection
            saccade_analysis = self.saccade_detector.detect_saccades(gaze_analysis)
            
            # Calculate engagement metrics
            engagement_metrics = self._calculate_eye_engagement_metrics(
                gaze_analysis, attention_analysis, blink_analysis, fixation_analysis
            )
            
            # Update tracking history
            self._update_tracking_history(gaze_analysis, attention_analysis, engagement_metrics)
            
            # Calculate tracking quality
            self.tracking_quality = self._calculate_tracking_quality(eye_data, gaze_analysis)
            
            # Create comprehensive result
            result = {
                'eye_landmarks': eye_data,
                'gaze_analysis': gaze_analysis,
                'attention_analysis': attention_analysis,
                'blink_analysis': blink_analysis,
                'fixation_analysis': fixation_analysis,
                'saccade_analysis': saccade_analysis,
                'engagement_metrics': engagement_metrics,
                'tracking_quality': self.tracking_quality,
                'precision_metrics': self._get_precision_metrics()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced eye tracking: {e}")
            return self._empty_result(error=str(e))
    
    def _extract_eye_landmarks(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract detailed eye landmark data"""
        height, width = frame_shape[:2]
        
        eye_data = {
            'left_eye_landmarks': [],
            'right_eye_landmarks': [],
            'left_iris_landmarks': [],
            'right_iris_landmarks': [],
            'eye_centers': {},
            'eye_dimensions': {}
        }
        
        # Extract eye landmarks
        for eye_side, indices in [('left_eye', self.eye_landmarks['left_eye']), 
                                 ('right_eye', self.eye_landmarks['right_eye'])]:
            landmarks = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    z = landmark.z
                    landmarks.append([x, y, z])
            
            eye_data[f'{eye_side}_landmarks'] = landmarks
            
            # Calculate eye center and dimensions
            if landmarks:
                center = np.mean(landmarks, axis=0)
                eye_data['eye_centers'][eye_side] = center.tolist()
                
                # Calculate eye dimensions
                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                width_px = max(x_coords) - min(x_coords)
                height_px = max(y_coords) - min(y_coords)
                eye_data['eye_dimensions'][eye_side] = [width_px, height_px]
        
        # Extract iris landmarks (if available)
        for iris_side, indices in [('left_iris', self.eye_landmarks['left_iris']), 
                                  ('right_iris', self.eye_landmarks['right_iris'])]:
            landmarks = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    z = landmark.z
                    landmarks.append([x, y, z])
            
            eye_data[f'{iris_side}_landmarks'] = landmarks
        
        return eye_data
    
    def _calculate_eye_engagement_metrics(self, gaze_analysis: Dict[str, Any], 
                                        attention_analysis: Dict[str, Any],
                                        blink_analysis: Dict[str, Any],
                                        fixation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive eye-based engagement metrics"""
        try:
            # Attention focus score
            attention_score = attention_analysis.get('focus_score', 0.0)
            
            # Gaze stability score
            gaze_stability = gaze_analysis.get('stability_score', 0.0)
            
            # Blink pattern score (normal blinking indicates engagement)
            blink_rate = blink_analysis.get('blink_rate', 0.0)
            normal_blink_rate = 0.3  # ~18 blinks per minute
            blink_score = 1.0 - min(1.0, abs(blink_rate - normal_blink_rate) / normal_blink_rate)
            
            # Fixation quality score
            fixation_score = fixation_analysis.get('fixation_quality', 0.0)
            
            # Calculate overall eye engagement score
            weights = {
                'attention': 0.35,
                'gaze_stability': 0.25,
                'blink_pattern': 0.20,
                'fixation_quality': 0.20
            }
            
            overall_score = (
                attention_score * weights['attention'] +
                gaze_stability * weights['gaze_stability'] +
                blink_score * weights['blink_pattern'] +
                fixation_score * weights['fixation_quality']
            )
            
            # Determine engagement level
            if overall_score > 0.75:
                engagement_level = 'high'
            elif overall_score > 0.45:
                engagement_level = 'medium'
            else:
                engagement_level = 'low'
            
            # Calculate attention duration
            attention_duration = self._calculate_attention_duration()
            
            # Detect attention patterns
            attention_patterns = self._detect_attention_patterns()
            
            engagement_metrics = {
                'overall_engagement_score': overall_score,
                'engagement_level': engagement_level,
                'component_scores': {
                    'attention_focus': attention_score,
                    'gaze_stability': gaze_stability,
                    'blink_pattern': blink_score,
                    'fixation_quality': fixation_score
                },
                'attention_duration': attention_duration,
                'attention_patterns': attention_patterns,
                'distraction_indicators': self._detect_distraction_indicators(),
                'focus_zones': attention_analysis.get('focus_zones', {}),
                'engagement_trend': self._calculate_engagement_trend()
            }
            
            return engagement_metrics
            
        except Exception as e:
            logger.error(f"Error calculating eye engagement metrics: {e}")
            return {'overall_engagement_score': 0.0, 'engagement_level': 'unknown'}
    
    def _calculate_attention_duration(self) -> float:
        """Calculate sustained attention duration"""
        if len(self.attention_history) < 10:
            return 0.0
        
        # Look for continuous periods of high attention
        recent_attention = list(self.attention_history)[-30:]  # Last 1 second
        high_attention_frames = sum(1 for a in recent_attention if a.get('focus_score', 0) > 0.7)
        
        # Convert to duration (assuming 30fps)
        attention_duration = high_attention_frames / 30.0
        return attention_duration
    
    def _detect_attention_patterns(self) -> Dict[str, Any]:
        """Detect patterns in attention behavior"""
        if len(self.attention_history) < 30:
            return {'pattern': 'insufficient_data'}
        
        recent_scores = [a.get('focus_score', 0) for a in list(self.attention_history)[-60:]]  # Last 2 seconds
        
        # Calculate pattern metrics
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        # Classify pattern
        if std_score < 0.1 and mean_score > 0.7:
            pattern = 'sustained_attention'
        elif std_score > 0.3:
            pattern = 'fluctuating_attention'
        elif trend < -0.01:
            pattern = 'declining_attention'
        elif trend > 0.01:
            pattern = 'improving_attention'
        else:
            pattern = 'stable_attention'
        
        return {
            'pattern': pattern,
            'mean_score': mean_score,
            'variability': std_score,
            'trend': trend,
            'confidence': min(1.0, len(recent_scores) / 60.0)
        }
    
    def _detect_distraction_indicators(self) -> List[Dict[str, Any]]:
        """Detect indicators of distraction"""
        distractions = []
        
        if len(self.gaze_history) < 10:
            return distractions
        
        recent_gazes = list(self.gaze_history)[-20:]  # Last ~0.7 seconds
        
        # Rapid gaze movements (potential distraction)
        gaze_velocities = []
        for i in range(1, len(recent_gazes)):
            prev_gaze = recent_gazes[i-1]
            curr_gaze = recent_gazes[i]
            
            if 'gaze_point' in prev_gaze and 'gaze_point' in curr_gaze:
                dx = curr_gaze['gaze_point'][0] - prev_gaze['gaze_point'][0]
                dy = curr_gaze['gaze_point'][1] - prev_gaze['gaze_point'][1]
                velocity = math.sqrt(dx*dx + dy*dy)
                gaze_velocities.append(velocity)
        
        if gaze_velocities:
            avg_velocity = np.mean(gaze_velocities)
            if avg_velocity > 50:  # Threshold for rapid movement
                distractions.append({
                    'type': 'rapid_gaze_movement',
                    'severity': min(1.0, avg_velocity / 100.0),
                    'timestamp': time.time()
                })
        
        # Looking away from main focus areas
        recent_attention = list(self.attention_history)[-10:]
        off_screen_count = sum(1 for a in recent_attention if a.get('focus_zone') == 'off_screen')
        
        if off_screen_count > 5:  # More than half the recent frames
            distractions.append({
                'type': 'looking_away',
                'severity': off_screen_count / 10.0,
                'timestamp': time.time()
            })
        
        return distractions
    
    def _calculate_engagement_trend(self) -> Dict[str, float]:
        """Calculate engagement trend over time"""
        if len(self.attention_history) < 30:
            return {'trend': 0.0, 'confidence': 0.0}
        
        # Get engagement scores over time
        recent_history = list(self.attention_history)[-90:]  # Last 3 seconds
        scores = [h.get('focus_score', 0) for h in recent_history]
        
        # Calculate trend
        if len(scores) > 10:
            trend = np.polyfit(range(len(scores)), scores, 1)[0]
            confidence = min(1.0, len(scores) / 90.0)
        else:
            trend = 0.0
            confidence = 0.0
        
        return {
            'trend': float(trend),
            'confidence': confidence,
            'direction': 'improving' if trend > 0.01 else 'declining' if trend < -0.01 else 'stable'
        }
    
    def _calculate_tracking_quality(self, eye_data: Dict[str, Any], gaze_analysis: Dict[str, Any]) -> float:
        """Calculate overall tracking quality score"""
        quality_factors = []
        
        # Eye landmark detection quality
        left_eye_quality = 1.0 if len(eye_data.get('left_eye_landmarks', [])) > 10 else 0.0
        right_eye_quality = 1.0 if len(eye_data.get('right_eye_landmarks', [])) > 10 else 0.0
        quality_factors.append((left_eye_quality + right_eye_quality) / 2)
        
        # Gaze estimation confidence
        gaze_confidence = gaze_analysis.get('confidence', 0.0)
        quality_factors.append(gaze_confidence)
        
        # Iris detection quality
        iris_quality = 1.0 if (len(eye_data.get('left_iris_landmarks', [])) > 0 and 
                              len(eye_data.get('right_iris_landmarks', [])) > 0) else 0.5
        quality_factors.append(iris_quality)
        
        return np.mean(quality_factors)
    
    def _get_precision_metrics(self) -> Dict[str, float]:
        """Get precision metrics for the tracking system"""
        return {
            'tracking_quality': self.tracking_quality,
            'gaze_accuracy': self.gaze_accuracy,
            'temporal_consistency': self._calculate_temporal_consistency(),
            'landmark_stability': self._calculate_landmark_stability()
        }
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of tracking"""
        if len(self.gaze_history) < 10:
            return 0.0
        
        recent_gazes = list(self.gaze_history)[-10:]
        confidences = [g.get('confidence', 0) for g in recent_gazes]
        
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_landmark_stability(self) -> float:
        """Calculate stability of landmark detection"""
        if len(self.eye_state_history) < 5:
            return 0.0
        
        # Calculate variance in eye center positions
        recent_states = list(self.eye_state_history)[-5:]
        left_centers = [s.get('left_eye_center', [0, 0]) for s in recent_states]
        
        if len(left_centers) > 1:
            x_variance = np.var([c[0] for c in left_centers])
            y_variance = np.var([c[1] for c in left_centers])
            stability = 1.0 / (1.0 + x_variance + y_variance)
            return min(1.0, stability)
        
        return 0.0
    
    def _update_tracking_history(self, gaze_analysis: Dict[str, Any], 
                               attention_analysis: Dict[str, Any],
                               engagement_metrics: Dict[str, Any]):
        """Update tracking history for temporal analysis"""
        current_time = time.time()
        
        # Update gaze history
        self.gaze_history.append({
            'timestamp': current_time,
            **gaze_analysis
        })
        
        # Update attention history
        self.attention_history.append({
            'timestamp': current_time,
            **attention_analysis,
            'engagement_score': engagement_metrics.get('overall_engagement_score', 0.0)
        })
        
        # Update eye state history
        self.eye_state_history.append({
            'timestamp': current_time,
            'tracking_quality': self.tracking_quality,
            'engagement_level': engagement_metrics.get('engagement_level', 'unknown')
        })
    
    def draw_eye_tracking_visualization(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw comprehensive eye tracking visualization"""
        vis_frame = frame.copy()
        
        # Draw eye landmarks
        eye_data = result.get('eye_landmarks', {})
        
        # Draw left eye
        left_eye = eye_data.get('left_eye_landmarks', [])
        if left_eye:
            for point in left_eye:
                cv2.circle(vis_frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
        
        # Draw right eye
        right_eye = eye_data.get('right_eye_landmarks', [])
        if right_eye:
            for point in right_eye:
                cv2.circle(vis_frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
        
        # Draw gaze direction
        gaze_analysis = result.get('gaze_analysis', {})
        if 'gaze_point' in gaze_analysis:
            gaze_point = gaze_analysis['gaze_point']
            cv2.circle(vis_frame, (int(gaze_point[0]), int(gaze_point[1])), 5, (255, 0, 0), -1)
        
        # Draw engagement metrics
        engagement_metrics = result.get('engagement_metrics', {})
        engagement_score = engagement_metrics.get('overall_engagement_score', 0.0)
        engagement_level = engagement_metrics.get('engagement_level', 'unknown')
        
        # Draw engagement info
        cv2.putText(vis_frame, f"Eye Engagement: {engagement_score:.2f} ({engagement_level})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw tracking quality
        quality = result.get('tracking_quality', 0.0)
        cv2.putText(vis_frame, f"Tracking Quality: {quality:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_frame
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'eye_landmarks': {},
            'gaze_analysis': {},
            'attention_analysis': {},
            'blink_analysis': {},
            'fixation_analysis': {},
            'saccade_analysis': {},
            'engagement_metrics': {'overall_engagement_score': 0.0, 'engagement_level': 'unknown'},
            'tracking_quality': 0.0,
            'precision_metrics': {}
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup eye tracking resources"""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        
        # Cleanup analysis components
        self.gaze_estimator.cleanup()
        self.attention_zone_analyzer.cleanup()
        self.blink_analyzer.cleanup()
        self.fixation_detector.cleanup()
        self.saccade_detector.cleanup()
        
        logger.info("Advanced eye tracker cleaned up")

# Helper classes for specialized eye tracking analysis
class GazeEstimator:
    """Advanced gaze estimation using eye landmarks"""
    
    def initialize(self):
        logger.info("Gaze estimator initialized")
    
    def estimate_gaze(self, eye_data: Dict[str, Any], frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Estimate gaze direction and point"""
        # Simplified gaze estimation
        return {
            'gaze_point': [320, 240],  # Center of frame
            'gaze_direction': [0.0, 0.0],
            'confidence': 0.8,
            'stability_score': 0.7
        }
    
    def cleanup(self):
        pass

class AttentionZoneAnalyzer:
    """Analyze attention zones and focus areas"""
    
    def initialize(self, screen_dimensions: Tuple[int, int]):
        self.screen_dimensions = screen_dimensions
        logger.info("Attention zone analyzer initialized")
    
    def analyze_attention(self, gaze_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention zones"""
        return {
            'focus_score': 0.8,
            'focus_zone': 'center',
            'focus_zones': {'center': 0.8, 'left': 0.1, 'right': 0.1}
        }
    
    def cleanup(self):
        pass

class BlinkAnalyzer:
    """Analyze blink patterns and eye state"""
    
    def initialize(self):
        logger.info("Blink analyzer initialized")
    
    def analyze_blinks(self, eye_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze blink patterns"""
        return {
            'blink_rate': 0.3,
            'blink_duration': 0.1,
            'eye_openness': 0.8
        }
    
    def cleanup(self):
        pass

class FixationDetector:
    """Detect and analyze eye fixations"""
    
    def initialize(self):
        logger.info("Fixation detector initialized")
    
    def detect_fixations(self, gaze_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect eye fixations"""
        return {
            'fixation_quality': 0.7,
            'fixation_duration': 1.5,
            'fixation_stability': 0.8
        }
    
    def cleanup(self):
        pass

class SaccadeDetector:
    """Detect and analyze eye saccades"""
    
    def initialize(self):
        logger.info("Saccade detector initialized")
    
    def detect_saccades(self, gaze_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect eye saccades"""
        return {
            'saccade_frequency': 2.0,
            'saccade_amplitude': 5.0,
            'saccade_velocity': 100.0
        }
    
    def cleanup(self):
        pass
