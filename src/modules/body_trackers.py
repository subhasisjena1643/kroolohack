"""
Specialized Body Part Trackers for Industry-Grade Movement Detection
Each tracker focuses on specific body parts with high precision analysis
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import time
import math
from collections import deque
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger

class BodyPartTracker:
    """Base class for body part tracking with precision metrics"""
    
    def __init__(self, part_name: str, smoothing_window: int = 5):
        self.part_name = part_name
        self.smoothing_window = smoothing_window
        self.position_history = deque(maxlen=100)
        self.velocity_history = deque(maxlen=50)
        self.acceleration_history = deque(maxlen=30)
        self.precision_score = 0.0
        
    def initialize(self):
        """Initialize tracker"""
        logger.info(f"Initialized {self.part_name} tracker")
    
    def get_precision(self) -> float:
        """Get tracking precision score"""
        return self.precision_score
    
    def cleanup(self):
        """Cleanup tracker resources"""
        pass

class EyeTracker(BodyPartTracker):
    """Advanced eye tracking for gaze analysis and attention detection"""
    
    def __init__(self, smoothing_window: int = 3):
        super().__init__("eyes", smoothing_window)
        
        # Eye landmark indices (MediaPipe face mesh)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.iris_indices = [468, 469, 470, 471, 472]  # Iris landmarks
        
        # Gaze tracking
        self.gaze_history = deque(maxlen=30)
        self.blink_detection = BlinkDetector()
        self.attention_zones = AttentionZoneAnalyzer()
        
    def analyze_movement(self, face_landmarks: List[List[float]]) -> Dict[str, Any]:
        """Analyze eye movement patterns for engagement detection"""
        try:
            if not face_landmarks or len(face_landmarks) < max(self.right_eye_indices):
                return {'error': 'Insufficient face landmarks'}
            
            # Extract eye regions
            left_eye_points = [face_landmarks[i][:3] for i in self.left_eye_indices if i < len(face_landmarks)]
            right_eye_points = [face_landmarks[i][:3] for i in self.right_eye_indices if i < len(face_landmarks)]
            
            # Calculate eye metrics
            eye_analysis = {
                'gaze_direction': self._calculate_gaze_direction(left_eye_points, right_eye_points),
                'eye_openness': self._calculate_eye_openness(left_eye_points, right_eye_points),
                'blink_analysis': self.blink_detection.analyze(left_eye_points, right_eye_points),
                'attention_focus': self.attention_zones.analyze_focus(left_eye_points, right_eye_points),
                'eye_movement_velocity': self._calculate_eye_velocity(),
                'fixation_stability': self._calculate_fixation_stability(),
                'engagement_indicators': self._calculate_eye_engagement_indicators()
            }
            
            # Update tracking history
            self._update_eye_history(eye_analysis)
            
            # Calculate precision
            self.precision_score = self._calculate_eye_precision(eye_analysis)
            
            return eye_analysis
            
        except Exception as e:
            logger.error(f"Error in eye movement analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_gaze_direction(self, left_eye: List, right_eye: List) -> Dict[str, float]:
        """Calculate precise gaze direction"""
        if not left_eye or not right_eye:
            return {'x': 0.0, 'y': 0.0, 'confidence': 0.0}
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Estimate gaze direction (simplified model)
        gaze_x = (left_center[0] + right_center[0]) / 2
        gaze_y = (left_center[1] + right_center[1]) / 2
        
        # Normalize to screen coordinates (-1 to 1)
        normalized_x = (gaze_x - 320) / 320  # Assuming 640px width
        normalized_y = (gaze_y - 240) / 240  # Assuming 480px height
        
        return {
            'x': float(normalized_x),
            'y': float(normalized_y),
            'confidence': 0.8  # Simplified confidence
        }
    
    def _calculate_eye_openness(self, left_eye: List, right_eye: List) -> Dict[str, float]:
        """Calculate eye openness ratio"""
        def eye_aspect_ratio(eye_points):
            if len(eye_points) < 6:
                return 0.0
            
            # Vertical distances
            A = euclidean(eye_points[1], eye_points[5])
            B = euclidean(eye_points[2], eye_points[4])
            
            # Horizontal distance
            C = euclidean(eye_points[0], eye_points[3])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C) if C > 0 else 0.0
            return ear
        
        left_ear = eye_aspect_ratio(left_eye) if left_eye else 0.0
        right_ear = eye_aspect_ratio(right_eye) if right_eye else 0.0
        
        return {
            'left_eye_openness': left_ear,
            'right_eye_openness': right_ear,
            'average_openness': (left_ear + right_ear) / 2,
            'asymmetry': abs(left_ear - right_ear)
        }
    
    def _calculate_eye_velocity(self) -> float:
        """Calculate eye movement velocity"""
        if len(self.gaze_history) < 2:
            return 0.0
        
        recent_gazes = list(self.gaze_history)[-5:]  # Last 5 frames
        velocities = []
        
        for i in range(1, len(recent_gazes)):
            prev_gaze = recent_gazes[i-1]['gaze_direction']
            curr_gaze = recent_gazes[i]['gaze_direction']
            
            dx = curr_gaze['x'] - prev_gaze['x']
            dy = curr_gaze['y'] - prev_gaze['y']
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0.0
    
    def _calculate_fixation_stability(self) -> float:
        """Calculate gaze fixation stability"""
        if len(self.gaze_history) < 10:
            return 0.0
        
        recent_gazes = list(self.gaze_history)[-10:]
        gaze_points = [(g['gaze_direction']['x'], g['gaze_direction']['y']) for g in recent_gazes]
        
        # Calculate standard deviation of gaze points
        x_coords = [p[0] for p in gaze_points]
        y_coords = [p[1] for p in gaze_points]
        
        stability = 1.0 / (1.0 + np.std(x_coords) + np.std(y_coords))
        return min(1.0, stability)
    
    def _calculate_eye_engagement_indicators(self) -> Dict[str, float]:
        """Calculate eye-based engagement indicators"""
        if len(self.gaze_history) < 5:
            return {'engagement_score': 0.0, 'attention_level': 'unknown'}
        
        recent_data = list(self.gaze_history)[-10:]
        
        # Calculate metrics
        avg_openness = np.mean([d.get('eye_openness', {}).get('average_openness', 0) for d in recent_data])
        blink_rate = np.mean([d.get('blink_analysis', {}).get('blink_rate', 0) for d in recent_data])
        fixation_stability = self._calculate_fixation_stability()
        
        # Engagement scoring
        openness_score = min(1.0, avg_openness * 3)  # Normal EAR is ~0.3
        blink_score = 1.0 - min(1.0, abs(blink_rate - 0.3) * 2)  # Normal blink rate ~0.3 Hz
        stability_score = fixation_stability
        
        engagement_score = (openness_score * 0.4 + blink_score * 0.3 + stability_score * 0.3)
        
        # Determine attention level
        if engagement_score > 0.7:
            attention_level = 'high'
        elif engagement_score > 0.4:
            attention_level = 'medium'
        else:
            attention_level = 'low'
        
        return {
            'engagement_score': engagement_score,
            'attention_level': attention_level,
            'component_scores': {
                'openness': openness_score,
                'blink_pattern': blink_score,
                'fixation_stability': stability_score
            }
        }
    
    def _update_eye_history(self, eye_analysis: Dict[str, Any]):
        """Update eye tracking history"""
        self.gaze_history.append({
            'timestamp': time.time(),
            **eye_analysis
        })
    
    def _calculate_eye_precision(self, eye_analysis: Dict[str, Any]) -> float:
        """Calculate eye tracking precision"""
        confidence = eye_analysis.get('gaze_direction', {}).get('confidence', 0.0)
        stability = eye_analysis.get('fixation_stability', 0.0)
        return (confidence + stability) / 2

class HandTracker(BodyPartTracker):
    """Advanced hand tracking for gesture and participation analysis"""
    
    def __init__(self, smoothing_window: int = 5):
        super().__init__("hands", smoothing_window)
        
        # Hand gesture patterns for engagement
        self.engagement_gestures = {
            'hand_raised': self._detect_hand_raised,
            'pointing': self._detect_pointing,
            'writing_motion': self._detect_writing_motion,
            'fidgeting': self._detect_fidgeting,
            'self_touch': self._detect_self_touch
        }
        
        self.hand_history = deque(maxlen=50)
        
    def analyze_movement(self, left_hand: List, right_hand: List) -> Dict[str, Any]:
        """Analyze hand movements for engagement patterns"""
        try:
            hand_analysis = {
                'left_hand_analysis': self._analyze_single_hand(left_hand, 'left'),
                'right_hand_analysis': self._analyze_single_hand(right_hand, 'right'),
                'bilateral_coordination': self._analyze_bilateral_coordination(left_hand, right_hand),
                'gesture_recognition': self._recognize_engagement_gestures(left_hand, right_hand),
                'movement_intensity': self._calculate_movement_intensity(),
                'engagement_indicators': self._calculate_hand_engagement_indicators()
            }
            
            # Update history
            self._update_hand_history(hand_analysis)
            
            # Calculate precision
            self.precision_score = self._calculate_hand_precision(hand_analysis)
            
            return hand_analysis
            
        except Exception as e:
            logger.error(f"Error in hand movement analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_single_hand(self, hand_landmarks: List, hand_side: str) -> Dict[str, Any]:
        """Analyze single hand movement patterns"""
        if not hand_landmarks:
            return {'present': False}
        
        # Calculate hand metrics
        hand_center = np.mean(hand_landmarks, axis=0)
        finger_positions = self._get_finger_positions(hand_landmarks)
        
        return {
            'present': True,
            'center_position': hand_center.tolist(),
            'finger_positions': finger_positions,
            'hand_openness': self._calculate_hand_openness(hand_landmarks),
            'movement_velocity': self._calculate_hand_velocity(hand_center, hand_side),
            'gesture_confidence': self._calculate_gesture_confidence(hand_landmarks)
        }
    
    def _get_finger_positions(self, hand_landmarks: List) -> Dict[str, List]:
        """Get individual finger positions"""
        if len(hand_landmarks) < 21:
            return {}
        
        finger_tips = {
            'thumb': hand_landmarks[4][:3],
            'index': hand_landmarks[8][:3],
            'middle': hand_landmarks[12][:3],
            'ring': hand_landmarks[16][:3],
            'pinky': hand_landmarks[20][:3]
        }
        
        return {k: [float(x) for x in v] for k, v in finger_tips.items()}
    
    def _calculate_hand_openness(self, hand_landmarks: List) -> float:
        """Calculate how open/closed the hand is"""
        if len(hand_landmarks) < 21:
            return 0.0
        
        # Calculate distances between fingertips and palm center
        palm_center = np.mean([hand_landmarks[0], hand_landmarks[5], hand_landmarks[17]], axis=0)
        fingertip_indices = [4, 8, 12, 16, 20]
        
        distances = []
        for tip_idx in fingertip_indices:
            distance = euclidean(hand_landmarks[tip_idx][:3], palm_center[:3])
            distances.append(distance)
        
        # Normalize openness (0 = closed fist, 1 = open hand)
        avg_distance = np.mean(distances)
        openness = min(1.0, avg_distance / 100.0)  # Normalize by typical hand size
        
        return openness
    
    def _calculate_hand_velocity(self, current_center: np.ndarray, hand_side: str) -> float:
        """Calculate hand movement velocity"""
        if len(self.hand_history) < 2:
            return 0.0
        
        # Get previous hand position
        prev_data = self.hand_history[-1]
        prev_analysis = prev_data.get(f'{hand_side}_hand_analysis', {})
        
        if not prev_analysis.get('present', False):
            return 0.0
        
        prev_center = np.array(prev_analysis['center_position'])
        velocity = euclidean(current_center, prev_center)
        
        return float(velocity)
    
    def _recognize_engagement_gestures(self, left_hand: List, right_hand: List) -> Dict[str, Any]:
        """Recognize engagement-related gestures"""
        recognized_gestures = {}
        
        for gesture_name, detector_func in self.engagement_gestures.items():
            try:
                confidence = detector_func(left_hand, right_hand)
                if confidence > 0.5:
                    recognized_gestures[gesture_name] = {
                        'confidence': confidence,
                        'detected': True
                    }
            except Exception as e:
                logger.error(f"Error detecting {gesture_name}: {e}")
        
        return recognized_gestures
    
    def _detect_hand_raised(self, left_hand: List, right_hand: List) -> float:
        """Detect hand raised gesture (high engagement indicator)"""
        max_confidence = 0.0
        
        for hand in [left_hand, right_hand]:
            if not hand or len(hand) < 21:
                continue
            
            wrist = hand[0]
            middle_tip = hand[12]
            
            # Check if hand is raised (fingertip above wrist)
            height_diff = wrist[1] - middle_tip[1]  # Y decreases upward
            
            if height_diff > 50:  # Significant height difference
                confidence = min(1.0, height_diff / 100.0)
                max_confidence = max(max_confidence, confidence)
        
        return max_confidence
    
    def _detect_writing_motion(self, left_hand: List, right_hand: List) -> float:
        """Detect writing motion (engagement indicator)"""
        # Simplified writing detection based on hand movement patterns
        if len(self.hand_history) < 10:
            return 0.0
        
        # Analyze movement patterns for writing-like motions
        # This would involve more complex analysis of hand trajectories
        return 0.0  # Placeholder
    
    def _detect_fidgeting(self, left_hand: List, right_hand: List) -> float:
        """Detect fidgeting behavior (potential disengagement indicator)"""
        if len(self.hand_history) < 5:
            return 0.0
        
        # Calculate movement variability
        recent_movements = []
        for data in list(self.hand_history)[-5:]:
            left_vel = data.get('left_hand_analysis', {}).get('movement_velocity', 0)
            right_vel = data.get('right_hand_analysis', {}).get('movement_velocity', 0)
            recent_movements.append(left_vel + right_vel)
        
        # High variability in small movements indicates fidgeting
        movement_std = np.std(recent_movements)
        movement_mean = np.mean(recent_movements)
        
        if movement_mean > 0:
            fidget_score = movement_std / movement_mean
            return min(1.0, fidget_score)
        
        return 0.0
    
    def _calculate_hand_engagement_indicators(self) -> Dict[str, float]:
        """Calculate hand-based engagement indicators"""
        if len(self.hand_history) < 3:
            return {'engagement_score': 0.0}
        
        recent_data = list(self.hand_history)[-5:]
        
        # Calculate engagement metrics
        gesture_activity = 0.0
        movement_purposefulness = 0.0
        
        for data in recent_data:
            gestures = data.get('gesture_recognition', {})
            positive_gestures = ['hand_raised', 'pointing', 'writing_motion']
            negative_gestures = ['fidgeting', 'self_touch']
            
            positive_score = sum([g.get('confidence', 0) for name, g in gestures.items() if name in positive_gestures])
            negative_score = sum([g.get('confidence', 0) for name, g in gestures.items() if name in negative_gestures])
            
            gesture_activity += positive_score - negative_score * 0.5
        
        gesture_activity = max(0.0, gesture_activity / len(recent_data))
        
        engagement_score = min(1.0, gesture_activity)
        
        return {
            'engagement_score': engagement_score,
            'gesture_activity': gesture_activity,
            'movement_purposefulness': movement_purposefulness
        }
    
    def _update_hand_history(self, hand_analysis: Dict[str, Any]):
        """Update hand tracking history"""
        self.hand_history.append({
            'timestamp': time.time(),
            **hand_analysis
        })
    
    def _calculate_hand_precision(self, hand_analysis: Dict[str, Any]) -> float:
        """Calculate hand tracking precision"""
        left_confidence = hand_analysis.get('left_hand_analysis', {}).get('gesture_confidence', 0.0)
        right_confidence = hand_analysis.get('right_hand_analysis', {}).get('gesture_confidence', 0.0)
        return (left_confidence + right_confidence) / 2

class PostureTracker(BodyPartTracker):
    """Advanced posture tracking for engagement analysis"""

    def __init__(self, smoothing_window: int = 10):
        super().__init__("posture", smoothing_window)

        # Key pose landmarks for posture analysis
        self.key_landmarks = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26
        }

        self.posture_history = deque(maxlen=30)
        self.baseline_posture = None

    def analyze_movement(self, pose_landmarks: List) -> Dict[str, Any]:
        """Analyze posture for engagement indicators"""
        try:
            if not pose_landmarks or len(pose_landmarks) < 27:
                return {'error': 'Insufficient pose landmarks'}

            # Extract key points
            key_points = {}
            for name, idx in self.key_landmarks.items():
                if idx < len(pose_landmarks):
                    key_points[name] = pose_landmarks[idx][:3]

            # Calculate posture metrics
            posture_analysis = {
                'spine_alignment': self._calculate_spine_alignment(key_points),
                'shoulder_level': self._calculate_shoulder_level(key_points),
                'head_position': self._calculate_head_position(key_points),
                'body_lean': self._calculate_body_lean(key_points),
                'posture_stability': self._calculate_posture_stability(),
                'engagement_posture': self._analyze_engagement_posture(key_points)
            }

            # Update history
            self._update_posture_history(posture_analysis)

            # Set baseline if not established
            if self.baseline_posture is None:
                self._establish_baseline_posture(posture_analysis)

            # Calculate precision
            self.precision_score = self._calculate_posture_precision(posture_analysis)

            return posture_analysis

        except Exception as e:
            logger.error(f"Error in posture analysis: {e}")
            return {'error': str(e)}

    def _calculate_spine_alignment(self, key_points: Dict) -> Dict[str, float]:
        """Calculate spine alignment metrics"""
        if 'nose' not in key_points or 'left_hip' not in key_points or 'right_hip' not in key_points:
            return {'alignment_score': 0.0}

        # Calculate hip center
        hip_center = np.mean([key_points['left_hip'], key_points['right_hip']], axis=0)

        # Calculate spine angle
        spine_vector = np.array(key_points['nose']) - hip_center
        vertical_vector = np.array([0, -1, 0])  # Upward direction

        # Calculate angle from vertical
        cos_angle = np.dot(spine_vector[:2], vertical_vector[:2]) / (np.linalg.norm(spine_vector[:2]) * np.linalg.norm(vertical_vector[:2]))
        angle = math.acos(np.clip(cos_angle, -1, 1))
        angle_degrees = math.degrees(angle)

        # Good alignment is close to 0 degrees
        alignment_score = max(0.0, 1.0 - angle_degrees / 30.0)  # Penalize deviation > 30 degrees

        return {
            'alignment_score': alignment_score,
            'spine_angle': angle_degrees,
            'deviation_from_vertical': angle_degrees
        }

    def _calculate_shoulder_level(self, key_points: Dict) -> Dict[str, float]:
        """Calculate shoulder level balance"""
        if 'left_shoulder' not in key_points or 'right_shoulder' not in key_points:
            return {'level_score': 0.0}

        left_shoulder = np.array(key_points['left_shoulder'])
        right_shoulder = np.array(key_points['right_shoulder'])

        # Calculate height difference
        height_diff = abs(left_shoulder[1] - right_shoulder[1])

        # Calculate level score (0 = very uneven, 1 = perfectly level)
        level_score = max(0.0, 1.0 - height_diff / 50.0)  # Normalize by typical shoulder width

        return {
            'level_score': level_score,
            'height_difference': float(height_diff),
            'tilt_direction': 'left' if left_shoulder[1] < right_shoulder[1] else 'right'
        }

    def _analyze_engagement_posture(self, key_points: Dict) -> Dict[str, float]:
        """Analyze posture for engagement indicators"""
        engagement_indicators = {}

        # Forward lean (interest indicator)
        if 'nose' in key_points and 'left_shoulder' in key_points and 'right_shoulder' in key_points:
            shoulder_center = np.mean([key_points['left_shoulder'], key_points['right_shoulder']], axis=0)
            forward_lean = key_points['nose'][2] - shoulder_center[2]  # Z-axis lean

            # Slight forward lean indicates engagement
            if forward_lean > 0:
                lean_score = min(1.0, forward_lean / 20.0)  # Normalize
            else:
                lean_score = max(0.0, 1.0 + forward_lean / 30.0)  # Backward lean is negative

            engagement_indicators['forward_lean_score'] = lean_score

        # Upright posture (attention indicator)
        spine_alignment = self._calculate_spine_alignment(key_points)
        engagement_indicators['upright_score'] = spine_alignment.get('alignment_score', 0.0)

        # Overall engagement posture score
        scores = [v for v in engagement_indicators.values() if isinstance(v, (int, float))]
        engagement_indicators['overall_engagement'] = np.mean(scores) if scores else 0.0

        return engagement_indicators

class MicroMovementTracker:
    """Tracker for micro-movements and subtle behavioral indicators"""

    def __init__(self):
        self.movement_history = deque(maxlen=100)
        self.micro_patterns = {}

    def initialize(self):
        """Initialize micro-movement tracker"""
        logger.info("Initialized micro-movement tracker")

    def analyze_movement(self, body_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze micro-movements for engagement patterns"""
        try:
            micro_analysis = {
                'head_micro_movements': self._analyze_head_micro_movements(body_data),
                'facial_micro_expressions': self._analyze_facial_micro_expressions(body_data),
                'breathing_patterns': self._analyze_breathing_patterns(body_data),
                'restlessness_indicators': self._analyze_restlessness(body_data),
                'attention_micro_signals': self._analyze_attention_micro_signals(body_data)
            }

            # Update history
            self._update_micro_history(micro_analysis)

            return micro_analysis

        except Exception as e:
            logger.error(f"Error in micro-movement analysis: {e}")
            return {'error': str(e)}

    def _analyze_head_micro_movements(self, body_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze subtle head movements"""
        face_landmarks = body_data.get('face_landmarks', [])

        if not face_landmarks or len(self.movement_history) < 5:
            return {'micro_movement_score': 0.0}

        # Calculate micro-movement patterns
        # This would involve analyzing small, rapid head movements
        # that might indicate confusion, disagreement, or disengagement

        return {'micro_movement_score': 0.0}  # Placeholder

    def _analyze_facial_micro_expressions(self, body_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze micro-expressions for emotional engagement"""
        # Placeholder for micro-expression analysis
        return {'micro_expression_score': 0.0}

    def _update_micro_history(self, micro_analysis: Dict[str, Any]):
        """Update micro-movement history"""
        self.movement_history.append({
            'timestamp': time.time(),
            **micro_analysis
        })

    def cleanup(self):
        """Cleanup micro-movement tracker"""
        pass

class DisengagementPatternRecognizer:
    """Intelligent pattern recognition for disengagement behaviors"""

    def __init__(self):
        self.pattern_history = deque(maxlen=50)
        self.disengagement_patterns = {
            'looking_away_pattern': self._detect_looking_away_pattern,
            'slouching_pattern': self._detect_slouching_pattern,
            'fidgeting_pattern': self._detect_fidgeting_pattern,
            'head_down_pattern': self._detect_head_down_pattern,
            'eye_closure_pattern': self._detect_eye_closure_pattern
        }

    def initialize(self):
        """Initialize pattern recognizer"""
        logger.info("Initialized disengagement pattern recognizer")

    def analyze_patterns(self, movement_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze movement patterns for disengagement indicators"""
        try:
            pattern_analysis = {
                'detected_patterns': [],
                'disengagement_score': 0.0,
                'pattern_confidence': 0.0,
                'pattern_details': {}
            }

            # Check each disengagement pattern
            total_confidence = 0.0
            detected_count = 0

            for pattern_name, detector_func in self.disengagement_patterns.items():
                confidence = detector_func(movement_analysis)

                if confidence > 0.6:  # High confidence threshold
                    pattern_analysis['detected_patterns'].append({
                        'pattern': pattern_name,
                        'confidence': confidence,
                        'severity': 'high' if confidence > 0.8 else 'medium'
                    })
                    detected_count += 1

                pattern_analysis['pattern_details'][pattern_name] = confidence
                total_confidence += confidence

            # Calculate overall disengagement score
            if detected_count > 0:
                pattern_analysis['disengagement_score'] = total_confidence / len(self.disengagement_patterns)
                pattern_analysis['pattern_confidence'] = total_confidence / detected_count

            # Update history
            self._update_pattern_history(pattern_analysis)

            return pattern_analysis

        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {'disengagement_score': 0.0, 'pattern_confidence': 0.0}

    def _detect_looking_away_pattern(self, movement_analysis: Dict[str, Any]) -> float:
        """Detect sustained looking away pattern"""
        eye_data = movement_analysis.get('eyes', {})
        gaze_direction = eye_data.get('gaze_direction', {})

        # Check if gaze is consistently away from center
        gaze_x = abs(gaze_direction.get('x', 0.0))
        gaze_y = abs(gaze_direction.get('y', 0.0))

        if gaze_x > 0.5 or gaze_y > 0.5:  # Looking significantly away
            return min(1.0, (gaze_x + gaze_y) / 1.0)

        return 0.0

    def _detect_slouching_pattern(self, movement_analysis: Dict[str, Any]) -> float:
        """Detect slouching posture pattern"""
        posture_data = movement_analysis.get('posture', {})
        spine_alignment = posture_data.get('spine_alignment', {})

        alignment_score = spine_alignment.get('alignment_score', 1.0)

        # Poor alignment indicates slouching
        slouch_score = 1.0 - alignment_score
        return slouch_score if slouch_score > 0.3 else 0.0

    def _update_pattern_history(self, pattern_analysis: Dict[str, Any]):
        """Update pattern recognition history"""
        self.pattern_history.append({
            'timestamp': time.time(),
            **pattern_analysis
        })

    def cleanup(self):
        """Cleanup pattern recognizer"""
        pass

# Helper classes for specialized analysis
class BlinkDetector:
    """Specialized blink detection and analysis"""

    def __init__(self):
        self.blink_history = deque(maxlen=30)

    def analyze(self, left_eye: List, right_eye: List) -> Dict[str, float]:
        """Analyze blink patterns"""
        # Simplified blink analysis
        return {'blink_rate': 0.3, 'blink_duration': 0.1}

class AttentionZoneAnalyzer:
    """Analyze attention focus zones"""

    def __init__(self):
        self.attention_zones = {
            'center': (0.0, 0.0, 0.3),  # Center zone
            'left': (-0.5, 0.0, 0.3),   # Left zone
            'right': (0.5, 0.0, 0.3),   # Right zone
            'up': (0.0, -0.5, 0.3),     # Up zone
            'down': (0.0, 0.5, 0.3)     # Down zone
        }

    def analyze_focus(self, left_eye: List, right_eye: List) -> Dict[str, Any]:
        """Analyze which zone the person is focusing on"""
        # Simplified focus analysis
        return {'primary_focus_zone': 'center', 'focus_confidence': 0.8}
