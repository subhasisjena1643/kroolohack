"""
Industry-Grade Advanced Body Movement Detection
SponsorLytix-level precision for detailed body movement analysis
Focuses on limbs, eyes, head, and micro-movements for engagement detection
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
import torch

from utils.base_processor import BaseProcessor
from utils.logger import logger

# Import specialized tracking classes
from modules.body_trackers import (
    BodyPartTracker, EyeTracker, HandTracker,
    PostureTracker, MicroMovementTracker, DisengagementPatternRecognizer
)

class AdvancedBodyDetector(BaseProcessor):
    """Industry-grade body movement detection with precision tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AdvancedBodyDetector", config)
        
        # MediaPipe Holistic for comprehensive body analysis
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = None
        
        # Detection parameters
        self.min_detection_confidence = config.get('min_detection_confidence', 0.7)
        self.min_tracking_confidence = config.get('min_tracking_confidence', 0.5)
        
        # Body part tracking
        self.body_landmarks_history = deque(maxlen=30)  # 1 second at 30fps
        self.movement_patterns = {}
        self.engagement_indicators = {}
        
        # Precision tracking for each body part
        self.head_tracker = BodyPartTracker("head", smoothing_window=5)
        self.eye_tracker = EyeTracker(smoothing_window=3)
        self.hand_tracker = HandTracker(smoothing_window=5)
        self.posture_tracker = PostureTracker(smoothing_window=10)
        self.micro_movement_tracker = MicroMovementTracker()
        
        # Engagement pattern recognition
        self.disengagement_patterns = DisengagementPatternRecognizer()
        
        # Performance optimization (ENHANCED FOR SPEED)
        self.frame_skip = config.get('frame_skip', 4)  # Process every 4th frame for speed
        self.frame_counter = 0
        
    def initialize(self) -> bool:
        """Initialize MediaPipe Holistic model"""
        try:
            logger.info("Initializing industry-grade body movement detection...")
            
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,  # OPTIMIZED: Fastest model (was 2)
                enable_segmentation=False,  # OPTIMIZED: Disable segmentation for speed
                refine_face_landmarks=False,  # OPTIMIZED: Disable refinement for speed
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            # Initialize all trackers
            self.head_tracker.initialize()
            self.eye_tracker.initialize()
            self.hand_tracker.initialize()
            self.posture_tracker.initialize()
            self.micro_movement_tracker.initialize()
            self.disengagement_patterns.initialize()
            
            logger.info("Advanced body detection initialized with industry-grade precision")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced body detector: {e}")
            return False
    
    def process_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for detailed body movement analysis"""
        try:
            self.frame_counter += 1
            
            # Frame skipping for performance optimization
            if self.frame_counter % self.frame_skip != 0:
                return self._get_cached_result()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Holistic
            results = self.holistic.process(rgb_frame)
            
            # Extract comprehensive body data
            body_data = self._extract_comprehensive_body_data(results, frame.shape)
            
            # Analyze movement patterns
            movement_analysis = self._analyze_movement_patterns(body_data)
            
            # Detect engagement/disengagement patterns
            engagement_analysis = self._analyze_engagement_patterns(movement_analysis)
            
            # Update tracking history
            self._update_tracking_history(body_data, movement_analysis)
            
            # Create comprehensive result
            result = {
                'body_landmarks': body_data,
                'movement_analysis': movement_analysis,
                'engagement_analysis': engagement_analysis,
                'precision_metrics': self._calculate_precision_metrics(),
                'confidence_scores': self._calculate_confidence_scores(results),
                'alert_triggers': self._check_alert_triggers(engagement_analysis)
            }
            
            # Cache result for frame skipping
            self._cache_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced body detection: {e}")
            return self._empty_result(error=str(e))
    
    def _extract_comprehensive_body_data(self, results, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract detailed body landmark data"""
        height, width = frame_shape[:2]
        
        body_data = {
            'face_landmarks': [],
            'pose_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'timestamp': time.time()
        }
        
        # Face landmarks (468 points for detailed analysis)
        if results.face_landmarks:
            face_points = []
            for landmark in results.face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                face_points.append([x, y, z, landmark.visibility])
            body_data['face_landmarks'] = face_points
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            pose_points = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                pose_points.append([x, y, z, landmark.visibility])
            body_data['pose_landmarks'] = pose_points
        
        # Hand landmarks (21 points each hand)
        if results.left_hand_landmarks:
            left_hand_points = []
            for landmark in results.left_hand_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                left_hand_points.append([x, y, z])
            body_data['left_hand_landmarks'] = left_hand_points
        
        if results.right_hand_landmarks:
            right_hand_points = []
            for landmark in results.right_hand_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                right_hand_points.append([x, y, z])
            body_data['right_hand_landmarks'] = right_hand_points
        
        return body_data
    
    def _analyze_movement_patterns(self, body_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed movement patterns for each body part"""
        movement_analysis = {}
        
        # Head movement analysis
        if body_data['face_landmarks']:
            head_analysis = self.head_tracker.analyze_movement(body_data['face_landmarks'])
            movement_analysis['head'] = head_analysis
        
        # Eye movement analysis
        if body_data['face_landmarks']:
            eye_analysis = self.eye_tracker.analyze_movement(body_data['face_landmarks'])
            movement_analysis['eyes'] = eye_analysis
        
        # Hand movement analysis
        hand_analysis = self.hand_tracker.analyze_movement(
            body_data['left_hand_landmarks'], 
            body_data['right_hand_landmarks']
        )
        movement_analysis['hands'] = hand_analysis
        
        # Posture analysis
        if body_data['pose_landmarks']:
            posture_analysis = self.posture_tracker.analyze_movement(body_data['pose_landmarks'])
            movement_analysis['posture'] = posture_analysis
        
        # Micro-movement analysis
        micro_analysis = self.micro_movement_tracker.analyze_movement(body_data)
        movement_analysis['micro_movements'] = micro_analysis
        
        return movement_analysis
    
    def _analyze_engagement_patterns(self, movement_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement patterns from movement data"""
        return self.disengagement_patterns.analyze_patterns(movement_analysis)
    
    def _calculate_precision_metrics(self) -> Dict[str, float]:
        """Calculate precision metrics for tracking quality"""
        return {
            'head_tracking_precision': self.head_tracker.get_precision(),
            'eye_tracking_precision': self.eye_tracker.get_precision(),
            'hand_tracking_precision': self.hand_tracker.get_precision(),
            'posture_tracking_precision': self.posture_tracker.get_precision(),
            'overall_precision': self._calculate_overall_precision()
        }
    
    def _calculate_confidence_scores(self, results) -> Dict[str, float]:
        """Calculate confidence scores for detections"""
        scores = {}
        
        if results.face_landmarks:
            face_confidences = [lm.visibility for lm in results.face_landmarks.landmark if hasattr(lm, 'visibility')]
            scores['face_confidence'] = np.mean(face_confidences) if face_confidences else 0.0
        
        if results.pose_landmarks:
            pose_confidences = [lm.visibility for lm in results.pose_landmarks.landmark if hasattr(lm, 'visibility')]
            scores['pose_confidence'] = np.mean(pose_confidences) if pose_confidences else 0.0
        
        scores['overall_confidence'] = np.mean(list(scores.values())) if scores else 0.0
        
        return scores
    
    def _check_alert_triggers(self, engagement_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert triggers based on engagement analysis"""
        alerts = []
        
        # Only trigger alerts for genuine disengagement patterns
        disengagement_score = engagement_analysis.get('disengagement_score', 0.0)
        confidence = engagement_analysis.get('pattern_confidence', 0.0)
        
        if disengagement_score > 0.7 and confidence > 0.8:
            alerts.append({
                'type': 'disengagement_detected',
                'severity': 'high' if disengagement_score > 0.9 else 'medium',
                'confidence': confidence,
                'patterns': engagement_analysis.get('detected_patterns', []),
                'timestamp': time.time()
            })
        
        return alerts
    
    def _update_tracking_history(self, body_data: Dict[str, Any], movement_analysis: Dict[str, Any]):
        """Update tracking history for temporal analysis"""
        self.body_landmarks_history.append({
            'timestamp': time.time(),
            'body_data': body_data,
            'movement_analysis': movement_analysis
        })
    
    def _calculate_overall_precision(self) -> float:
        """Calculate overall tracking precision"""
        precisions = [
            self.head_tracker.get_precision(),
            self.eye_tracker.get_precision(),
            self.hand_tracker.get_precision(),
            self.posture_tracker.get_precision()
        ]
        return np.mean([p for p in precisions if p > 0])
    
    def _get_cached_result(self) -> Dict[str, Any]:
        """Get cached result for frame skipping"""
        return getattr(self, '_cached_result', self._empty_result())
    
    def _cache_result(self, result: Dict[str, Any]):
        """Cache result for frame skipping"""
        self._cached_result = result
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'body_landmarks': {},
            'movement_analysis': {},
            'engagement_analysis': {'disengagement_score': 0.0, 'pattern_confidence': 0.0},
            'precision_metrics': {},
            'confidence_scores': {},
            'alert_triggers': []
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def draw_advanced_annotations(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw advanced annotations on frame"""
        annotated_frame = frame.copy()
        
        # Draw precision metrics
        precision = result.get('precision_metrics', {})
        y_offset = 30
        for metric, value in precision.items():
            text = f"{metric}: {value:.3f}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        # Draw alert indicators
        alerts = result.get('alert_triggers', [])
        if alerts:
            for i, alert in enumerate(alerts):
                alert_text = f"ALERT: {alert['type']} ({alert['confidence']:.2f})"
                cv2.putText(annotated_frame, alert_text, (10, frame.shape[0] - 50 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated_frame
    
    def cleanup(self):
        """Cleanup resources"""
        if self.holistic:
            self.holistic.close()
            self.holistic = None
        
        # Cleanup all trackers
        self.head_tracker.cleanup()
        self.eye_tracker.cleanup()
        self.hand_tracker.cleanup()
        self.posture_tracker.cleanup()
        self.micro_movement_tracker.cleanup()
        self.disengagement_patterns.cleanup()
        
        logger.info("Advanced body detector cleaned up")
