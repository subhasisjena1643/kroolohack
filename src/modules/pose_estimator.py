"""
Head Pose Estimation Module using MediaPipe
Detects attention direction and engagement through head pose
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional
import math

from utils.base_processor import BaseProcessor
from utils.logger import logger

class HeadPoseEstimator(BaseProcessor):
    """MediaPipe-based head pose estimation for attention detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("HeadPoseEstimator", config)
        
        # MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = None
        
        # Pose estimation parameters
        self.confidence_threshold = config.get('pose_confidence_threshold', 0.5)
        self.attention_angle_threshold = config.get('attention_angle_threshold', 30.0)
        
        # 3D model points for pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera matrix (will be calculated)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Attention tracking
        self.attention_history = []
        self.current_attention_scores = {}
    
    def initialize(self) -> bool:
        """Initialize MediaPipe face mesh"""
        try:
            logger.info("Initializing MediaPipe face mesh for pose estimation...")
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=self.confidence_threshold
            )
            
            logger.info("Head pose estimator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize head pose estimator: {e}")
            return False
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process frame and face data for head pose estimation"""
        try:
            frame = data.get('frame')
            faces = data.get('faces', [])
            
            if frame is None:
                return {'poses': [], 'attention_scores': {}}
            
            # Setup camera matrix if not done
            if self.camera_matrix is None:
                self._setup_camera_matrix(frame.shape)
            
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            # Extract pose information
            poses = []
            attention_scores = {}
            
            if results.multi_face_landmarks:
                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    pose_data = self._estimate_head_pose(face_landmarks, frame.shape)
                    if pose_data:
                        poses.append(pose_data)
                        
                        # Calculate attention score
                        attention_score = self._calculate_attention_score(pose_data)
                        attention_scores[f'face_{i}'] = attention_score
            
            # Update attention history
            self._update_attention_history(attention_scores)
            
            result = {
                'poses': poses,
                'attention_scores': attention_scores,
                'average_attention': self._get_average_attention(),
                'attention_distribution': self._get_attention_distribution()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in head pose estimation: {e}")
            return {
                'poses': [],
                'attention_scores': {},
                'error': str(e)
            }
    
    def _setup_camera_matrix(self, frame_shape: Tuple[int, int, int]):
        """Setup camera matrix for pose estimation"""
        height, width = frame_shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
    
    def _estimate_head_pose(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        """Estimate head pose from face landmarks"""
        try:
            height, width = frame_shape[:2]
            
            # Extract key facial landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            
            # Key points for pose estimation (nose tip, chin, eye corners, mouth corners)
            key_indices = [1, 152, 33, 263, 61, 291]  # MediaPipe face mesh indices
            
            if len(landmarks) < max(key_indices):
                return None
            
            image_points = np.array([
                landmarks[key_indices[0]],  # Nose tip
                landmarks[key_indices[1]],  # Chin
                landmarks[key_indices[2]],  # Left eye left corner
                landmarks[key_indices[3]],  # Right eye right corner
                landmarks[key_indices[4]],  # Left mouth corner
                landmarks[key_indices[5]]   # Right mouth corner
            ], dtype="double")
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                return None
            
            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Calculate face center
            face_center = np.mean(image_points, axis=0).astype(int)
            
            pose_data = {
                'rotation_vector': rotation_vector.flatten().tolist(),
                'translation_vector': translation_vector.flatten().tolist(),
                'euler_angles': angles,
                'face_center': face_center.tolist(),
                'landmarks': landmarks.tolist(),
                'key_points': image_points.tolist()
            }
            
            return pose_data
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return None
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Dict[str, float]:
        """Convert rotation matrix to Euler angles"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        return {
            'pitch': math.degrees(x),  # Up/down
            'yaw': math.degrees(y),    # Left/right
            'roll': math.degrees(z)    # Tilt
        }
    
    def _calculate_attention_score(self, pose_data: Dict[str, Any]) -> float:
        """Calculate attention score based on head pose"""
        angles = pose_data['euler_angles']
        
        # Get absolute angles
        abs_pitch = abs(angles['pitch'])
        abs_yaw = abs(angles['yaw'])
        abs_roll = abs(angles['roll'])
        
        # Calculate attention score (0-1, where 1 is fully attentive)
        # Looking straight ahead gets highest score
        yaw_score = max(0, 1 - (abs_yaw / 45.0))  # Penalize left/right movement
        pitch_score = max(0, 1 - (abs_pitch / 30.0))  # Penalize up/down movement
        roll_score = max(0, 1 - (abs_roll / 20.0))  # Penalize head tilt
        
        # Weighted combination
        attention_score = (yaw_score * 0.5 + pitch_score * 0.3 + roll_score * 0.2)
        
        # Determine attention state
        if attention_score > 0.7:
            attention_state = 'high'
        elif attention_score > 0.4:
            attention_state = 'medium'
        else:
            attention_state = 'low'
        
        return {
            'score': attention_score,
            'state': attention_state,
            'angles': angles,
            'component_scores': {
                'yaw': yaw_score,
                'pitch': pitch_score,
                'roll': roll_score
            }
        }
    
    def _update_attention_history(self, attention_scores: Dict[str, Any]):
        """Update attention history for trend analysis"""
        import time
        current_time = time.time()
        
        self.attention_history.append({
            'timestamp': current_time,
            'scores': attention_scores
        })
        
        # Keep only last 30 seconds of history
        cutoff_time = current_time - 30.0
        self.attention_history = [h for h in self.attention_history if h['timestamp'] > cutoff_time]
    
    def _get_average_attention(self) -> float:
        """Get average attention score across all faces"""
        if not self.attention_history:
            return 0.0
        
        recent_scores = []
        for entry in self.attention_history[-10:]:  # Last 10 entries
            for face_id, score_data in entry['scores'].items():
                recent_scores.append(score_data['score'])
        
        return sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
    
    def _get_attention_distribution(self) -> Dict[str, float]:
        """Get distribution of attention states"""
        if not self.attention_history:
            return {'high': 0.0, 'medium': 0.0, 'low': 0.0}
        
        state_counts = {'high': 0, 'medium': 0, 'low': 0}
        total_count = 0
        
        for entry in self.attention_history[-20:]:  # Last 20 entries
            for face_id, score_data in entry['scores'].items():
                state_counts[score_data['state']] += 1
                total_count += 1
        
        if total_count == 0:
            return {'high': 0.0, 'medium': 0.0, 'low': 0.0}
        
        return {
            state: count / total_count 
            for state, count in state_counts.items()
        }
    
    def draw_pose_estimation(self, frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw pose estimation results on frame"""
        result_frame = frame.copy()
        
        for pose in poses:
            try:
                # Draw face center
                center = pose['face_center']
                cv2.circle(result_frame, tuple(center), 5, (0, 255, 255), -1)
                
                # Draw pose axes
                angles = pose['euler_angles']
                
                # Project 3D axes onto image
                axis_points = np.array([
                    [0, 0, 0],
                    [50, 0, 0],  # X-axis (red)
                    [0, 50, 0],  # Y-axis (green)
                    [0, 0, -50]  # Z-axis (blue)
                ], dtype=np.float32)
                
                rotation_vector = np.array(pose['rotation_vector'])
                translation_vector = np.array(pose['translation_vector'])
                
                projected_points, _ = cv2.projectPoints(
                    axis_points,
                    rotation_vector,
                    translation_vector,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                projected_points = projected_points.reshape(-1, 2).astype(int)
                
                # Draw axes
                origin = tuple(projected_points[0])
                cv2.arrowedLine(result_frame, origin, tuple(projected_points[1]), (0, 0, 255), 3)  # X - Red
                cv2.arrowedLine(result_frame, origin, tuple(projected_points[2]), (0, 255, 0), 3)  # Y - Green
                cv2.arrowedLine(result_frame, origin, tuple(projected_points[3]), (255, 0, 0), 3)  # Z - Blue
                
                # Draw angle information
                text_y = center[1] - 60
                cv2.putText(result_frame, f"Pitch: {angles['pitch']:.1f}°", 
                           (center[0] - 50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(result_frame, f"Yaw: {angles['yaw']:.1f}°", 
                           (center[0] - 50, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(result_frame, f"Roll: {angles['roll']:.1f}°", 
                           (center[0] - 50, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                logger.error(f"Error drawing pose: {e}")
        
        return result_frame
    
    def cleanup(self):
        """Cleanup resources"""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        logger.info("Head pose estimator cleaned up")
