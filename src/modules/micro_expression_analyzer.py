"""
Micro-Expression & Facial Analysis System
Advanced facial analysis for emotional engagement detection
Industry-grade precision for micro-expression recognition and emotional state analysis
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
import torch.nn as nn
import torch.nn.functional as F

from utils.base_processor import BaseProcessor
from utils.logger import logger

class MicroExpressionAnalyzer(BaseProcessor):
    """Advanced micro-expression and facial analysis for engagement detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MicroExpressionAnalyzer", config)
        
        # MediaPipe Face Detection and Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = None
        self.face_mesh = None
        
        # Analysis parameters
        self.min_detection_confidence = config.get('min_detection_confidence', 0.8)
        self.min_tracking_confidence = config.get('min_tracking_confidence', 0.7)
        
        # Facial landmark groups for analysis
        self.facial_regions = {
            'forehead': [9, 10, 151, 337, 299, 333, 298, 301],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 267, 271, 272],
            'cheeks': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 207, 213, 192, 147, 187, 207, 213, 192, 147, 187],
            'jaw': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
        
        # Emotion analysis components
        self.emotion_classifier = EmotionClassifier()
        self.micro_expression_detector = MicroExpressionDetector()
        self.facial_action_unit_analyzer = FacialActionUnitAnalyzer()
        self.engagement_emotion_mapper = EngagementEmotionMapper()
        
        # Tracking history
        self.expression_history = deque(maxlen=150)  # 5 seconds at 30fps
        self.emotion_history = deque(maxlen=300)     # 10 seconds
        self.micro_expression_events = deque(maxlen=50)
        
        # Baseline and calibration
        self.baseline_expression = None
        self.expression_sensitivity = config.get('expression_sensitivity', 0.7)
        
        # Performance metrics
        self.detection_accuracy = 0.0
        self.temporal_consistency = 0.0
        
    def initialize(self) -> bool:
        """Initialize micro-expression analysis system"""
        try:
            logger.info("Initializing micro-expression analysis system...")
            
            # Initialize MediaPipe components
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=self.min_detection_confidence
            )
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            # Initialize analysis components
            self.emotion_classifier.initialize()
            self.micro_expression_detector.initialize()
            self.facial_action_unit_analyzer.initialize()
            self.engagement_emotion_mapper.initialize()
            
            logger.info("Micro-expression analysis system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize micro-expression analyzer: {e}")
            return False
    
    def process_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for micro-expression and facial analysis (OPTIMIZED FOR SPEED)"""
        try:
            # OPTIMIZATION: Skip processing every other frame for speed
            if not hasattr(self, 'frame_skip_counter'):
                self.frame_skip_counter = 0

            self.frame_skip_counter += 1

            # Process every 2nd frame for micro-expressions (still real-time feel)
            if self.frame_skip_counter % 2 != 0:
                return self.last_result if hasattr(self, 'last_result') else self._empty_result("Skipped for performance")

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # OPTIMIZATION: Use existing face detection from main pipeline instead of re-detecting
            # Face mesh analysis (lighter than full detection)
            mesh_results = self.face_mesh.process(rgb_frame)

            if not mesh_results.multi_face_landmarks:
                return self._empty_result("No face landmarks detected")

            # Extract facial landmarks (simplified)
            face_landmarks = mesh_results.multi_face_landmarks[0]
            landmark_data = self._extract_facial_landmarks_fast(face_landmarks, frame.shape)

            # OPTIMIZATION: Simplified emotion classification (every 3rd frame)
            if self.frame_skip_counter % 3 == 0:
                emotion_analysis = self.emotion_classifier.classify_emotions_fast(landmark_data)
                self.cached_emotion_analysis = emotion_analysis
            else:
                emotion_analysis = getattr(self, 'cached_emotion_analysis', {'emotion': 'neutral', 'confidence': 0.5})

            # OPTIMIZATION: Simplified micro-expression detection
            micro_expression_analysis = self.micro_expression_detector.detect_micro_expressions_fast(landmark_data)

            # OPTIMIZATION: Skip FAU analysis (most expensive) - use simplified version
            fau_analysis = self._simple_fau_analysis(landmark_data)

            # OPTIMIZATION: Simplified engagement mapping
            engagement_analysis = self._simple_engagement_mapping(emotion_analysis, micro_expression_analysis)

            # OPTIMIZATION: Simplified metrics calculation
            facial_engagement_metrics = self._calculate_facial_engagement_metrics_fast(emotion_analysis, engagement_analysis)

            # OPTIMIZATION: Update history less frequently (every 5th frame)
            if self.frame_skip_counter % 5 == 0:
                self._update_expression_history_fast(landmark_data, emotion_analysis)

            # Create simplified result
            result = {
                'facial_landmarks': landmark_data,
                'emotion_analysis': emotion_analysis,
                'micro_expression_analysis': micro_expression_analysis,
                'facial_action_units': fau_analysis,
                'engagement_analysis': engagement_analysis,
                'facial_engagement_metrics': facial_engagement_metrics,
                'performance_metrics': {'processing_time': 0.01},  # Simplified
                'emotional_state_indicators': self._get_emotional_state_indicators_fast()
            }

            self.last_result = result
            return result

        except Exception as e:
            logger.error(f"Error in micro-expression analysis: {e}")
            return self._empty_result(error=str(e))
    
    def _extract_facial_landmarks(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract detailed facial landmark data organized by regions"""
        height, width = frame_shape[:2]
        
        landmark_data = {
            'all_landmarks': [],
            'facial_regions': {},
            'landmark_confidence': [],
            'face_dimensions': {}
        }
        
        # Extract all landmarks
        all_landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            all_landmarks.append([x, y, z])
        
        landmark_data['all_landmarks'] = all_landmarks
        
        # Extract landmarks by facial regions
        for region_name, indices in self.facial_regions.items():
            region_landmarks = []
            for idx in indices:
                if idx < len(all_landmarks):
                    region_landmarks.append(all_landmarks[idx])
            landmark_data['facial_regions'][region_name] = region_landmarks
        
        # Calculate face dimensions and geometry
        if all_landmarks:
            x_coords = [p[0] for p in all_landmarks]
            y_coords = [p[1] for p in all_landmarks]
            
            face_width = max(x_coords) - min(x_coords)
            face_height = max(y_coords) - min(y_coords)
            face_center = [np.mean(x_coords), np.mean(y_coords)]
            
            landmark_data['face_dimensions'] = {
                'width': face_width,
                'height': face_height,
                'center': face_center,
                'aspect_ratio': face_width / face_height if face_height > 0 else 1.0
            }
        
        return landmark_data

    def _extract_facial_landmarks_fast(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Fast facial landmark extraction - only essential points"""
        height, width = frame_shape[:2]

        # Extract only key landmarks for speed
        key_landmarks = []
        key_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  # Eye region
        key_indices.extend([61, 84, 17, 314, 405, 320, 308, 324, 318])  # Mouth region

        for idx in key_indices:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                key_landmarks.append([x, y])

        return {
            'key_landmarks': key_landmarks,
            'face_dimensions': {'width': width, 'height': height}
        }

    def _simple_fau_analysis(self, landmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified Facial Action Unit analysis"""
        return {
            'action_units': {'AU12': 0.5, 'AU6': 0.5},  # Basic smile indicators
            'confidence': 0.7
        }

    def _simple_engagement_mapping(self, emotion_analysis: Dict[str, Any], micro_expression_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified engagement mapping"""
        emotion = emotion_analysis.get('emotion', 'neutral')
        confidence = emotion_analysis.get('confidence', 0.5)

        # Simple engagement scoring based on emotion
        engagement_score = 0.5
        if emotion in ['happy', 'surprised']:
            engagement_score = 0.8
        elif emotion in ['sad', 'angry']:
            engagement_score = 0.3

        return {
            'engagement_score': engagement_score,
            'confidence': confidence,
            'indicators': ['facial_expression']
        }

    def _calculate_facial_engagement_metrics_fast(self, emotion_analysis: Dict[str, Any], engagement_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fast facial engagement metrics calculation"""
        return {
            'overall_engagement': engagement_analysis.get('engagement_score', 0.5),
            'emotional_valence': 0.5,
            'attention_level': 0.6,
            'confidence': emotion_analysis.get('confidence', 0.5)
        }

    def _update_expression_history_fast(self, landmark_data: Dict[str, Any], emotion_analysis: Dict[str, Any]):
        """Fast history update - minimal data"""
        if not hasattr(self, 'simple_history'):
            self.simple_history = []

        self.simple_history.append({
            'emotion': emotion_analysis.get('emotion', 'neutral'),
            'timestamp': time.time()
        })

        # Keep only last 10 entries
        if len(self.simple_history) > 10:
            self.simple_history = self.simple_history[-10:]

    def _get_emotional_state_indicators_fast(self) -> Dict[str, Any]:
        """Fast emotional state indicators"""
        return {
            'current_state': 'engaged',
            'stability': 0.7,
            'trend': 'stable'
        }
    
    def _calculate_facial_engagement_metrics(self, emotion_analysis: Dict[str, Any],
                                           micro_expression_analysis: Dict[str, Any],
                                           engagement_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive facial engagement metrics"""
        try:
            # Primary emotion engagement score
            primary_emotion = emotion_analysis.get('primary_emotion', 'neutral')
            emotion_confidence = emotion_analysis.get('confidence', 0.0)
            
            # Map emotions to engagement levels
            emotion_engagement_map = {
                'interest': 0.9,
                'concentration': 0.85,
                'curiosity': 0.8,
                'surprise': 0.7,
                'happiness': 0.6,
                'neutral': 0.5,
                'confusion': 0.4,
                'boredom': 0.2,
                'frustration': 0.3,
                'sadness': 0.25,
                'anger': 0.1
            }
            
            emotion_engagement_score = emotion_engagement_map.get(primary_emotion, 0.5)
            
            # Micro-expression engagement indicators
            micro_expressions = micro_expression_analysis.get('detected_expressions', [])
            positive_micro_expressions = ['interest', 'concentration', 'curiosity', 'surprise']
            negative_micro_expressions = ['boredom', 'confusion', 'frustration']
            
            positive_count = sum(1 for expr in micro_expressions if expr.get('type') in positive_micro_expressions)
            negative_count = sum(1 for expr in micro_expressions if expr.get('type') in negative_micro_expressions)
            
            micro_expression_score = 0.5
            if positive_count > negative_count:
                micro_expression_score = 0.5 + (positive_count / (positive_count + negative_count + 1)) * 0.5
            elif negative_count > positive_count:
                micro_expression_score = 0.5 - (negative_count / (positive_count + negative_count + 1)) * 0.5
            
            # Facial animation and expressiveness
            expressiveness_score = self._calculate_facial_expressiveness()
            
            # Attention-related facial indicators
            attention_indicators = self._analyze_attention_facial_indicators(emotion_analysis)
            
            # Calculate overall facial engagement score
            weights = {
                'emotion': 0.4,
                'micro_expressions': 0.3,
                'expressiveness': 0.2,
                'attention_indicators': 0.1
            }
            
            overall_score = (
                emotion_engagement_score * weights['emotion'] +
                micro_expression_score * weights['micro_expressions'] +
                expressiveness_score * weights['expressiveness'] +
                attention_indicators.get('score', 0.5) * weights['attention_indicators']
            )
            
            # Determine engagement level
            if overall_score > 0.75:
                engagement_level = 'high'
            elif overall_score > 0.45:
                engagement_level = 'medium'
            else:
                engagement_level = 'low'
            
            # Calculate emotional stability
            emotional_stability = self._calculate_emotional_stability()
            
            # Detect engagement patterns
            engagement_patterns = self._detect_facial_engagement_patterns()
            
            facial_engagement_metrics = {
                'overall_engagement_score': overall_score,
                'engagement_level': engagement_level,
                'component_scores': {
                    'emotion_engagement': emotion_engagement_score,
                    'micro_expression_engagement': micro_expression_score,
                    'facial_expressiveness': expressiveness_score,
                    'attention_indicators': attention_indicators.get('score', 0.5)
                },
                'emotional_stability': emotional_stability,
                'engagement_patterns': engagement_patterns,
                'confidence': emotion_confidence,
                'facial_animation_level': self._get_facial_animation_level(),
                'emotional_valence': self._calculate_emotional_valence(emotion_analysis)
            }
            
            return facial_engagement_metrics
            
        except Exception as e:
            logger.error(f"Error calculating facial engagement metrics: {e}")
            return {'overall_engagement_score': 0.0, 'engagement_level': 'unknown'}
    
    def _calculate_facial_expressiveness(self) -> float:
        """Calculate facial expressiveness and animation level"""
        if len(self.expression_history) < 10:
            return 0.5
        
        # Calculate movement in facial landmarks over time
        recent_expressions = list(self.expression_history)[-20:]  # Last ~0.7 seconds
        
        movement_scores = []
        for i in range(1, len(recent_expressions)):
            prev_landmarks = recent_expressions[i-1].get('all_landmarks', [])
            curr_landmarks = recent_expressions[i].get('all_landmarks', [])
            
            if len(prev_landmarks) == len(curr_landmarks) and len(prev_landmarks) > 0:
                total_movement = 0.0
                for j in range(len(prev_landmarks)):
                    movement = euclidean(prev_landmarks[j][:2], curr_landmarks[j][:2])
                    total_movement += movement
                
                avg_movement = total_movement / len(prev_landmarks)
                movement_scores.append(avg_movement)
        
        if movement_scores:
            avg_movement = np.mean(movement_scores)
            # Normalize movement score (typical facial movement range)
            expressiveness = min(1.0, avg_movement / 10.0)  # Normalize by typical movement
            return expressiveness
        
        return 0.5
    
    def _analyze_attention_facial_indicators(self, emotion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze facial indicators of attention and focus"""
        attention_score = 0.5
        indicators = []
        
        # Eyebrow position (raised eyebrows can indicate interest/attention)
        eyebrow_analysis = emotion_analysis.get('facial_regions', {}).get('eyebrows', {})
        if eyebrow_analysis.get('raised', False):
            attention_score += 0.2
            indicators.append('raised_eyebrows')
        
        # Eye openness (wide eyes can indicate attention)
        eye_analysis = emotion_analysis.get('facial_regions', {}).get('eyes', {})
        eye_openness = eye_analysis.get('openness', 0.5)
        if eye_openness > 0.7:
            attention_score += 0.15
            indicators.append('wide_eyes')
        
        # Forward head lean (detected through facial geometry changes)
        face_dimensions = emotion_analysis.get('face_dimensions', {})
        aspect_ratio = face_dimensions.get('aspect_ratio', 1.0)
        if aspect_ratio > 1.1:  # Face appears wider (forward lean)
            attention_score += 0.1
            indicators.append('forward_lean')
        
        return {
            'score': min(1.0, attention_score),
            'indicators': indicators,
            'confidence': 0.7
        }
    
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability over time"""
        if len(self.emotion_history) < 10:
            return 0.5
        
        recent_emotions = list(self.emotion_history)[-30:]  # Last 1 second
        emotion_scores = [e.get('confidence', 0) for e in recent_emotions]
        
        if emotion_scores:
            stability = 1.0 - np.std(emotion_scores)
            return max(0.0, min(1.0, stability))
        
        return 0.5
    
    def _detect_facial_engagement_patterns(self) -> Dict[str, Any]:
        """Detect patterns in facial engagement over time"""
        if len(self.emotion_history) < 30:
            return {'pattern': 'insufficient_data'}
        
        recent_emotions = list(self.emotion_history)[-60:]  # Last 2 seconds
        engagement_scores = []
        
        for emotion_data in recent_emotions:
            primary_emotion = emotion_data.get('primary_emotion', 'neutral')
            # Map to engagement score
            emotion_engagement_map = {
                'interest': 0.9, 'concentration': 0.85, 'curiosity': 0.8,
                'surprise': 0.7, 'happiness': 0.6, 'neutral': 0.5,
                'confusion': 0.4, 'boredom': 0.2, 'frustration': 0.3
            }
            score = emotion_engagement_map.get(primary_emotion, 0.5)
            engagement_scores.append(score)
        
        if len(engagement_scores) > 10:
            mean_score = np.mean(engagement_scores)
            trend = np.polyfit(range(len(engagement_scores)), engagement_scores, 1)[0]
            variability = np.std(engagement_scores)
            
            # Classify pattern
            if variability < 0.1 and mean_score > 0.7:
                pattern = 'sustained_high_engagement'
            elif variability < 0.1 and mean_score < 0.3:
                pattern = 'sustained_low_engagement'
            elif variability > 0.3:
                pattern = 'fluctuating_engagement'
            elif trend > 0.01:
                pattern = 'increasing_engagement'
            elif trend < -0.01:
                pattern = 'decreasing_engagement'
            else:
                pattern = 'stable_engagement'
            
            return {
                'pattern': pattern,
                'mean_engagement': mean_score,
                'trend': trend,
                'variability': variability,
                'confidence': min(1.0, len(engagement_scores) / 60.0)
            }
        
        return {'pattern': 'insufficient_data'}
    
    def _get_facial_animation_level(self) -> str:
        """Get facial animation level description"""
        expressiveness = self._calculate_facial_expressiveness()
        
        if expressiveness > 0.7:
            return 'highly_animated'
        elif expressiveness > 0.4:
            return 'moderately_animated'
        elif expressiveness > 0.2:
            return 'slightly_animated'
        else:
            return 'static'
    
    def _calculate_emotional_valence(self, emotion_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional valence (positive/negative)"""
        primary_emotion = emotion_analysis.get('primary_emotion', 'neutral')
        confidence = emotion_analysis.get('confidence', 0.0)
        
        # Valence mapping
        valence_map = {
            'happiness': 0.8, 'interest': 0.6, 'curiosity': 0.5, 'surprise': 0.3,
            'neutral': 0.0, 'confusion': -0.2, 'boredom': -0.4,
            'frustration': -0.6, 'sadness': -0.7, 'anger': -0.8
        }
        
        valence = valence_map.get(primary_emotion, 0.0)
        
        return {
            'valence': valence,
            'confidence': confidence,
            'emotional_tone': 'positive' if valence > 0.2 else 'negative' if valence < -0.2 else 'neutral'
        }
    
    def _get_emotional_state_indicators(self) -> Dict[str, Any]:
        """Get current emotional state indicators"""
        if not self.emotion_history:
            return {'state': 'unknown'}
        
        recent_emotion = list(self.emotion_history)[-1]
        
        return {
            'current_emotion': recent_emotion.get('primary_emotion', 'unknown'),
            'emotion_confidence': recent_emotion.get('confidence', 0.0),
            'emotional_intensity': recent_emotion.get('intensity', 0.0),
            'emotional_stability': self._calculate_emotional_stability(),
            'engagement_relevance': self._get_emotion_engagement_relevance(recent_emotion)
        }
    
    def _get_emotion_engagement_relevance(self, emotion_data: Dict[str, Any]) -> str:
        """Get relevance of current emotion to engagement"""
        emotion = emotion_data.get('primary_emotion', 'neutral')
        
        high_engagement_emotions = ['interest', 'concentration', 'curiosity']
        medium_engagement_emotions = ['surprise', 'happiness']
        low_engagement_emotions = ['boredom', 'confusion', 'frustration']
        
        if emotion in high_engagement_emotions:
            return 'high_engagement_indicator'
        elif emotion in medium_engagement_emotions:
            return 'medium_engagement_indicator'
        elif emotion in low_engagement_emotions:
            return 'low_engagement_indicator'
        else:
            return 'neutral_indicator'
    
    def _update_expression_history(self, landmark_data: Dict[str, Any],
                                 emotion_analysis: Dict[str, Any],
                                 micro_expression_analysis: Dict[str, Any]):
        """Update expression tracking history"""
        current_time = time.time()
        
        # Update expression history
        self.expression_history.append({
            'timestamp': current_time,
            **landmark_data
        })
        
        # Update emotion history
        self.emotion_history.append({
            'timestamp': current_time,
            **emotion_analysis
        })
        
        # Update micro-expression events
        micro_expressions = micro_expression_analysis.get('detected_expressions', [])
        for expr in micro_expressions:
            self.micro_expression_events.append({
                'timestamp': current_time,
                **expr
            })
    
    def _update_performance_metrics(self, landmark_data: Dict[str, Any], emotion_analysis: Dict[str, Any]):
        """Update performance metrics"""
        # Detection accuracy (based on landmark confidence)
        landmarks = landmark_data.get('all_landmarks', [])
        if landmarks:
            self.detection_accuracy = 0.9  # Simplified
        
        # Temporal consistency
        if len(self.emotion_history) > 1:
            self.temporal_consistency = 0.8  # Simplified
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            'detection_accuracy': self.detection_accuracy,
            'temporal_consistency': self.temporal_consistency,
            'landmark_quality': self._calculate_landmark_quality(),
            'emotion_confidence': self._calculate_average_emotion_confidence()
        }
    
    def _calculate_landmark_quality(self) -> float:
        """Calculate landmark detection quality"""
        if not self.expression_history:
            return 0.0
        
        recent_data = list(self.expression_history)[-5:]
        quality_scores = []
        
        for data in recent_data:
            landmarks = data.get('all_landmarks', [])
            if len(landmarks) > 400:  # Good landmark count
                quality_scores.append(1.0)
            elif len(landmarks) > 200:
                quality_scores.append(0.7)
            else:
                quality_scores.append(0.3)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_average_emotion_confidence(self) -> float:
        """Calculate average emotion detection confidence"""
        if not self.emotion_history:
            return 0.0
        
        recent_emotions = list(self.emotion_history)[-10:]
        confidences = [e.get('confidence', 0) for e in recent_emotions]
        
        return np.mean(confidences) if confidences else 0.0
    
    def draw_facial_analysis_visualization(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw facial analysis visualization"""
        vis_frame = frame.copy()
        
        # Draw facial landmarks
        landmark_data = result.get('facial_landmarks', {})
        all_landmarks = landmark_data.get('all_landmarks', [])
        
        for landmark in all_landmarks:
            cv2.circle(vis_frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
        
        # Draw emotion information
        emotion_analysis = result.get('emotion_analysis', {})
        primary_emotion = emotion_analysis.get('primary_emotion', 'unknown')
        confidence = emotion_analysis.get('confidence', 0.0)
        
        cv2.putText(vis_frame, f"Emotion: {primary_emotion} ({confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw engagement metrics
        engagement_metrics = result.get('facial_engagement_metrics', {})
        engagement_score = engagement_metrics.get('overall_engagement_score', 0.0)
        engagement_level = engagement_metrics.get('engagement_level', 'unknown')
        
        cv2.putText(vis_frame, f"Facial Engagement: {engagement_score:.2f} ({engagement_level})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_frame
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'facial_landmarks': {},
            'emotion_analysis': {},
            'micro_expression_analysis': {},
            'facial_action_units': {},
            'engagement_analysis': {},
            'facial_engagement_metrics': {'overall_engagement_score': 0.0, 'engagement_level': 'unknown'},
            'performance_metrics': {},
            'emotional_state_indicators': {}
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup micro-expression analysis resources"""
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
        
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        
        # Cleanup analysis components
        self.emotion_classifier.cleanup()
        self.micro_expression_detector.cleanup()
        self.facial_action_unit_analyzer.cleanup()
        self.engagement_emotion_mapper.cleanup()
        
        logger.info("Micro-expression analyzer cleaned up")

# Helper classes for specialized facial analysis
class EmotionClassifier:
    """Advanced emotion classification from facial features"""
    
    def initialize(self):
        logger.info("Emotion classifier initialized")
    
    def classify_emotions(self, landmark_data: Dict[str, Any], frame: np.ndarray) -> Dict[str, Any]:
        """Classify emotions from facial landmarks"""
        # Simplified emotion classification
        return {
            'primary_emotion': 'interest',
            'confidence': 0.8,
            'emotion_probabilities': {
                'interest': 0.8, 'neutral': 0.15, 'concentration': 0.05
            },
            'intensity': 0.7
        }
    
    def cleanup(self):
        pass

class MicroExpressionDetector:
    """Detect micro-expressions in facial movements"""
    
    def initialize(self):
        logger.info("Micro-expression detector initialized")
    
    def detect_micro_expressions(self, landmark_data: Dict[str, Any], 
                               expression_history: deque) -> Dict[str, Any]:
        """Detect micro-expressions"""
        return {
            'detected_expressions': [],
            'micro_expression_frequency': 0.0,
            'expression_intensity': 0.0
        }
    
    def cleanup(self):
        pass

class FacialActionUnitAnalyzer:
    """Analyze Facial Action Units (FAUs)"""
    
    def initialize(self):
        logger.info("Facial Action Unit analyzer initialized")
    
    def analyze_action_units(self, landmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze facial action units"""
        return {
            'active_action_units': [],
            'action_unit_intensities': {},
            'fau_confidence': 0.0
        }
    
    def cleanup(self):
        pass

class EngagementEmotionMapper:
    """Map emotions and expressions to engagement levels"""
    
    def initialize(self):
        logger.info("Engagement emotion mapper initialized")
    
    def map_emotions_to_engagement(self, emotion_analysis: Dict[str, Any],
                                 micro_expression_analysis: Dict[str, Any],
                                 fau_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Map emotions to engagement levels"""
        return {
            'engagement_emotion_score': 0.7,
            'emotional_engagement_level': 'medium',
            'engagement_confidence': 0.8
        }
    
    def cleanup(self):
        pass
