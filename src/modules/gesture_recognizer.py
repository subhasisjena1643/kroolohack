"""
Hand Gesture Recognition Module using MediaPipe
Detects participation gestures like hand raising, pointing, etc.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional
import time
from collections import deque

from utils.base_processor import BaseProcessor
from utils.logger import logger

class GestureRecognizer(BaseProcessor):
    """MediaPipe-based hand gesture recognition for participation tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("GestureRecognizer", config)
        
        # MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        
        # Gesture recognition parameters
        self.confidence_threshold = config.get('gesture_confidence_threshold', 0.7)
        self.buffer_frames = config.get('gesture_buffer_frames', 10)
        
        # Gesture tracking
        self.gesture_history = deque(maxlen=self.buffer_frames)
        self.participation_events = []
        self.current_gestures = {}
        
        # Gesture definitions
        self.gesture_definitions = {
            'hand_raised': self._detect_hand_raised,
            'pointing': self._detect_pointing,
            'thumbs_up': self._detect_thumbs_up,
            'open_palm': self._detect_open_palm,
            'fist': self._detect_fist
        }
        
        # Participation scoring
        self.participation_score = 0.0
        self.gesture_counts = {gesture: 0 for gesture in self.gesture_definitions.keys()}
    
    def initialize(self) -> bool:
        """Initialize MediaPipe hands detection"""
        try:
            logger.info("Initializing MediaPipe hands for gesture recognition...")
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=10,  # Classroom setting
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=self.confidence_threshold
            )
            
            logger.info("Gesture recognizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize gesture recognizer: {e}")
            return False
    
    def process_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for hand gesture recognition"""
        try:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Extract hand information
            hands_data = []
            detected_gestures = {}
            
            if results.multi_hand_landmarks:
                for i, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    hand_data = self._extract_hand_data(hand_landmarks, handedness, frame.shape)
                    hands_data.append(hand_data)
                    
                    # Recognize gestures for this hand
                    gestures = self._recognize_gestures(hand_data)
                    detected_gestures[f'hand_{i}'] = gestures
            
            # Update gesture history
            self._update_gesture_history(detected_gestures)
            
            # Calculate participation metrics
            participation_metrics = self._calculate_participation_metrics()
            
            result = {
                'hands': hands_data,
                'gestures': detected_gestures,
                'stable_gestures': self._get_stable_gestures(),
                'participation_events': self._get_recent_participation_events(),
                'participation_score': self.participation_score,
                'gesture_counts': self.gesture_counts.copy(),
                'participation_metrics': participation_metrics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in gesture recognition: {e}")
            return {
                'hands': [],
                'gestures': {},
                'stable_gestures': {},
                'participation_events': [],
                'error': str(e)
            }
    
    def _extract_hand_data(self, hand_landmarks, handedness, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract hand data from MediaPipe landmarks"""
        height, width = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            landmarks.append([x, y, z])
        
        landmarks = np.array(landmarks)
        
        # Calculate hand bounding box
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        bbox = [int(np.min(x_coords)), int(np.min(y_coords)), 
                int(np.max(x_coords)), int(np.max(y_coords))]
        
        # Calculate hand center and size
        center = [int(np.mean(x_coords)), int(np.mean(y_coords))]
        hand_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        # Hand orientation (left/right)
        hand_label = handedness.classification[0].label
        hand_score = handedness.classification[0].score
        
        hand_data = {
            'landmarks': landmarks.tolist(),
            'bbox': bbox,
            'center': center,
            'size': hand_size,
            'hand_type': hand_label,
            'confidence': hand_score,
            'wrist_position': landmarks[0].tolist(),  # Wrist landmark
            'fingertips': self._get_fingertip_positions(landmarks)
        }
        
        return hand_data
    
    def _get_fingertip_positions(self, landmarks: np.ndarray) -> Dict[str, List[int]]:
        """Get fingertip positions"""
        # MediaPipe hand landmark indices for fingertips
        fingertip_indices = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        fingertips = {}
        for finger, index in fingertip_indices.items():
            fingertips[finger] = landmarks[index][:2].astype(int).tolist()
        
        return fingertips
    
    def _recognize_gestures(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize gestures from hand data"""
        gestures = {}
        landmarks = np.array(hand_data['landmarks'])
        
        for gesture_name, detector_func in self.gesture_definitions.items():
            try:
                confidence = detector_func(landmarks, hand_data)
                if confidence > 0.5:  # Threshold for gesture detection
                    gestures[gesture_name] = {
                        'confidence': confidence,
                        'detected': True
                    }
            except Exception as e:
                logger.error(f"Error detecting {gesture_name}: {e}")
        
        return gestures
    
    def _detect_hand_raised(self, landmarks: np.ndarray, hand_data: Dict[str, Any]) -> float:
        """Detect hand raised gesture (participation indicator)"""
        wrist = landmarks[0]
        middle_finger_tip = landmarks[12]
        
        # Check if hand is raised (fingertip significantly above wrist)
        height_diff = wrist[1] - middle_finger_tip[1]  # Y decreases upward
        hand_size = hand_data['size']
        
        # Normalize by hand size
        relative_height = height_diff / hand_size if hand_size > 0 else 0
        
        # Hand is raised if fingertip is well above wrist
        if relative_height > 1.5:  # Threshold for raised hand
            confidence = min(1.0, relative_height / 2.0)
            return confidence
        
        return 0.0
    
    def _detect_pointing(self, landmarks: np.ndarray, hand_data: Dict[str, Any]) -> float:
        """Detect pointing gesture"""
        # Index finger extended, others curled
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Check if index finger is extended
        index_extended = (index_tip[1] < index_pip[1])  # Tip above PIP joint
        
        # Check if middle finger is curled
        middle_curled = (middle_tip[1] > middle_pip[1])  # Tip below PIP joint
        
        if index_extended and middle_curled:
            return 0.8
        elif index_extended:
            return 0.5
        
        return 0.0
    
    def _detect_thumbs_up(self, landmarks: np.ndarray, hand_data: Dict[str, Any]) -> float:
        """Detect thumbs up gesture"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        
        # Thumb extended upward
        thumb_up = (thumb_tip[1] < thumb_ip[1])
        
        # Other fingers curled (check index as representative)
        index_curled = (index_tip[1] > index_pip[1])
        
        if thumb_up and index_curled:
            return 0.9
        elif thumb_up:
            return 0.6
        
        return 0.0
    
    def _detect_open_palm(self, landmarks: np.ndarray, hand_data: Dict[str, Any]) -> float:
        """Detect open palm gesture"""
        fingertip_indices = [4, 8, 12, 16, 20]  # All fingertips
        pip_indices = [3, 6, 10, 14, 18]  # PIP joints
        
        extended_count = 0
        for tip_idx, pip_idx in zip(fingertip_indices, pip_indices):
            if landmarks[tip_idx][1] < landmarks[pip_idx][1]:  # Finger extended
                extended_count += 1
        
        # Open palm if most fingers are extended
        confidence = extended_count / 5.0
        return confidence if confidence > 0.6 else 0.0
    
    def _detect_fist(self, landmarks: np.ndarray, hand_data: Dict[str, Any]) -> float:
        """Detect fist gesture"""
        fingertip_indices = [8, 12, 16, 20]  # Exclude thumb
        pip_indices = [6, 10, 14, 18]
        
        curled_count = 0
        for tip_idx, pip_idx in zip(fingertip_indices, pip_indices):
            if landmarks[tip_idx][1] > landmarks[pip_idx][1]:  # Finger curled
                curled_count += 1
        
        # Fist if most fingers are curled
        confidence = curled_count / 4.0
        return confidence if confidence > 0.7 else 0.0
    
    def _update_gesture_history(self, detected_gestures: Dict[str, Any]):
        """Update gesture history for stability analysis"""
        current_time = time.time()
        
        self.gesture_history.append({
            'timestamp': current_time,
            'gestures': detected_gestures
        })
        
        # Update participation events
        for hand_id, gestures in detected_gestures.items():
            for gesture_name, gesture_data in gestures.items():
                if gesture_data['detected'] and gesture_data['confidence'] > 0.7:
                    # Check if this is a new gesture event
                    if self._is_new_gesture_event(gesture_name, current_time):
                        self.participation_events.append({
                            'timestamp': current_time,
                            'gesture': gesture_name,
                            'hand': hand_id,
                            'confidence': gesture_data['confidence']
                        })
                        
                        # Update gesture counts
                        self.gesture_counts[gesture_name] += 1
                        
                        logger.info(f"Participation gesture detected: {gesture_name} ({gesture_data['confidence']:.2f})")
        
        # Clean old participation events (keep last 5 minutes)
        cutoff_time = current_time - 300.0
        self.participation_events = [e for e in self.participation_events if e['timestamp'] > cutoff_time]
    
    def _is_new_gesture_event(self, gesture_name: str, current_time: float) -> bool:
        """Check if this is a new gesture event (not a continuation)"""
        # Look for recent events of the same gesture
        recent_events = [e for e in self.participation_events 
                        if e['gesture'] == gesture_name and current_time - e['timestamp'] < 2.0]
        
        return len(recent_events) == 0
    
    def _get_stable_gestures(self) -> Dict[str, Any]:
        """Get gestures that are stable across multiple frames"""
        if len(self.gesture_history) < 3:
            return {}
        
        stable_gestures = {}
        recent_frames = list(self.gesture_history)[-5:]  # Last 5 frames
        
        # Count gesture occurrences across frames
        gesture_counts = {}
        for frame in recent_frames:
            for hand_id, gestures in frame['gestures'].items():
                for gesture_name in gestures.keys():
                    key = f"{hand_id}_{gesture_name}"
                    gesture_counts[key] = gesture_counts.get(key, 0) + 1
        
        # Consider gestures stable if they appear in most recent frames
        stability_threshold = len(recent_frames) * 0.6
        for key, count in gesture_counts.items():
            if count >= stability_threshold:
                hand_id, gesture_name = key.split('_', 1)
                if hand_id not in stable_gestures:
                    stable_gestures[hand_id] = {}
                stable_gestures[hand_id][gesture_name] = {
                    'stability': count / len(recent_frames),
                    'frames_detected': count
                }
        
        return stable_gestures
    
    def _calculate_participation_metrics(self) -> Dict[str, Any]:
        """Calculate participation metrics"""
        current_time = time.time()
        
        # Recent participation (last 2 minutes)
        recent_events = [e for e in self.participation_events 
                        if current_time - e['timestamp'] < 120.0]
        
        # Calculate participation score
        participation_gestures = ['hand_raised', 'pointing', 'thumbs_up']
        recent_participation = [e for e in recent_events 
                              if e['gesture'] in participation_gestures]
        
        # Update participation score (exponential moving average)
        current_participation = len(recent_participation) / 10.0  # Normalize
        self.participation_score = 0.7 * self.participation_score + 0.3 * current_participation
        self.participation_score = min(1.0, self.participation_score)
        
        return {
            'recent_events_count': len(recent_events),
            'participation_events_count': len(recent_participation),
            'participation_rate': len(recent_participation) / 120.0,  # Events per second
            'most_common_gesture': self._get_most_common_gesture(),
            'engagement_level': self._get_engagement_level()
        }
    
    def _get_recent_participation_events(self) -> List[Dict[str, Any]]:
        """Get recent participation events"""
        current_time = time.time()
        return [e for e in self.participation_events 
                if current_time - e['timestamp'] < 30.0]  # Last 30 seconds
    
    def _get_most_common_gesture(self) -> str:
        """Get the most commonly detected gesture"""
        if not self.gesture_counts:
            return "none"
        
        return max(self.gesture_counts, key=self.gesture_counts.get)
    
    def _get_engagement_level(self) -> str:
        """Get overall engagement level based on participation"""
        if self.participation_score > 0.7:
            return "high"
        elif self.participation_score > 0.3:
            return "medium"
        else:
            return "low"
    
    def draw_gestures(self, frame: np.ndarray, hands_data: List[Dict[str, Any]], 
                     gestures: Dict[str, Any]) -> np.ndarray:
        """Draw gesture recognition results on frame"""
        result_frame = frame.copy()
        
        for i, hand_data in enumerate(hands_data):
            hand_id = f'hand_{i}'
            
            # Draw hand landmarks
            landmarks = np.array(hand_data['landmarks'])
            for landmark in landmarks:
                cv2.circle(result_frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)
            
            # Draw bounding box
            bbox = hand_data['bbox']
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            
            # Draw detected gestures
            if hand_id in gestures:
                y_offset = bbox[1] - 10
                for gesture_name, gesture_data in gestures[hand_id].items():
                    if gesture_data['detected']:
                        text = f"{gesture_name}: {gesture_data['confidence']:.2f}"
                        cv2.putText(result_frame, text, (bbox[0], y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_offset -= 15
        
        # Draw participation info
        participation_text = f"Participation Score: {self.participation_score:.2f}"
        cv2.putText(result_frame, participation_text, (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def cleanup(self):
        """Cleanup resources"""
        if self.hands:
            self.hands.close()
            self.hands = None
        logger.info("Gesture recognizer cleaned up")
