"""
Face Detection Module
Basic face detection for engagement analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import mediapipe as mp
import time

from utils.base_processor import BaseProcessor
from utils.logger import logger

class FaceDetector(BaseProcessor):
    """Basic face detection for engagement analysis"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("FaceDetector", config)

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Face tracking
        self.face_count = 0
        self.face_positions = []

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for face detection"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.face_detection.process(rgb_frame)

            faces = []
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    faces.append({
                        'bbox': [x, y, width, height],
                        'confidence': detection.score[0],
                        'center': [x + width//2, y + height//2]
                    })

            self.face_count = len(faces)
            self.face_positions = [face['center'] for face in faces]

            return {
                'faces': faces,
                'face_count': self.face_count,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return {
                'faces': [],
                'face_count': 0,
                'timestamp': time.time()
            }

    def get_face_count(self) -> int:
        """Get current face count"""
        return self.face_count

    def get_face_positions(self) -> List[List[int]]:
        """Get current face positions"""
        return self.face_positions

    def initialize(self) -> bool:
        """Initialize the face detector"""
        try:
            logger.info("Face detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            return False

    def process_data(self, data: Any) -> Dict[str, Any]:
        """Process data (alias for process_frame)"""
        if isinstance(data, np.ndarray):
            return self.process_frame(data)
        else:
            logger.error("Invalid data type for face detection")
            return {'faces': [], 'face_count': 0, 'timestamp': time.time()}

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            logger.info("Face detector cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during face detector cleanup: {e}")

    def draw_detections(self, frame: np.ndarray, detections: Any) -> np.ndarray:
        """Draw face detection results on frame"""
        try:
            # Handle different detection result formats
            if isinstance(detections, dict):
                faces = detections.get('faces', [])
            elif isinstance(detections, list):
                faces = detections
            else:
                return frame

            for face in faces:
                # Handle different face data formats
                if isinstance(face, dict):
                    bbox = face.get('bbox', face.get('bounding_box', []))
                    confidence = face.get('confidence', face.get('detection_confidence', 0.0))
                else:
                    # Skip if face data format is unexpected
                    continue

                if len(bbox) >= 4:
                    # Draw bounding box
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    # Ensure coordinates are valid
                    x = max(0, min(x, frame.shape[1] - 1))
                    y = max(0, min(y, frame.shape[0] - 1))
                    w = max(1, min(w, frame.shape[1] - x))
                    h = max(1, min(h, frame.shape[0] - y))

                    # Draw green rectangle for face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw confidence score
                    label = f"Face: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    # Draw background for text
                    cv2.rectangle(frame, (x, y - label_size[1] - 10),
                                (x + label_size[0], y), (0, 255, 0), -1)

                    # Draw text
                    cv2.putText(frame, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            return frame

        except Exception as e:
            logger.error(f"Error drawing face detections: {e}")
            return frame

class AdvancedBodyDetector(BaseProcessor):
    """Industry-grade body movement detection for engagement analysis"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("AdvancedBodyDetector", config)

        # Multi-model approach for precision
        self.yolo_model = None
        self.pose_model = None
        self.face_mesh_model = None
        self.hands_model = None

        # Detection parameters
        self.confidence_threshold = config.get('face_confidence_threshold', 0.7)  # Higher for industry grade
        self.max_persons = config.get('max_faces', 30)
        self.movement_sensitivity = config.get('movement_sensitivity', 0.02)

        # MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Movement tracking
        self.person_trackers = {}  # Track individual persons
        self.movement_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.engagement_indicators = {}

        # Performance optimization
        self.input_size = (1280, 720)  # Higher resolution for precision
        self.device = 'cpu'

        # Industry-grade metrics
        self.precision_metrics = {
            'detection_accuracy': 0.0,
            'tracking_stability': 0.0,
            'movement_precision': 0.0
        }
    
    def initialize(self) -> bool:
        """Initialize YOLOv8 face detection model"""
        try:
            logger.info("Loading YOLOv8 face detection model...")
            
            # Load YOLOv8 model (will download if not present)
            self.model = YOLO('yolov8n.pt')  # Using general YOLOv8 for person detection
            
            # Warm up the model
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            
            logger.info("YOLOv8 face detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            return False
    
    def process_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for face detection and attendance"""
        try:
            # Resize frame for processing
            processed_frame = cv2.resize(frame, self.input_size)
            
            # Run YOLOv8 detection
            results = self.model(processed_frame, verbose=False, conf=self.confidence_threshold)
            
            # Extract face detections
            faces = self._extract_faces(results[0], frame.shape)
            
            # Update attendance
            self._update_attendance(faces)
            
            # Create result
            result = {
                'faces': faces,
                'face_count': len(faces),
                'attendance_count': self.attendance_count,
                'face_positions': self.face_positions,
                'frame_shape': frame.shape
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return {
                'faces': [],
                'face_count': 0,
                'attendance_count': 0,
                'face_positions': [],
                'error': str(e)
            }
    
    def _extract_faces(self, detection_result, original_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Extract face information from YOLOv8 results"""
        faces = []
        
        if detection_result.boxes is None:
            return faces
        
        # Scale factors for coordinate conversion
        scale_x = original_shape[1] / self.input_size[0]
        scale_y = original_shape[0] / self.input_size[1]
        
        boxes = detection_result.boxes.xyxy.cpu().numpy()
        confidences = detection_result.boxes.conf.cpu().numpy()
        classes = detection_result.boxes.cls.cpu().numpy()
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            # Filter for person class (class 0 in COCO dataset)
            if int(cls) == 0 and conf >= self.confidence_threshold:
                # Scale coordinates back to original frame size
                x1, y1, x2, y2 = box
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Calculate face center and size
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                
                face_info = {
                    'id': i,
                    'bbox': [x1, y1, x2, y2],
                    'center': [center_x, center_y],
                    'size': [width, height],
                    'confidence': float(conf),
                    'area': width * height
                }
                
                faces.append(face_info)
        
        # Sort by confidence and limit to max_faces
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        return faces[:self.max_faces]
    
    def _update_attendance(self, faces: List[Dict[str, Any]]):
        """Update attendance count based on detected faces"""
        current_time = time.time()
        
        # Simple attendance tracking based on face count
        # In a real system, you'd use face tracking/recognition
        current_face_count = len(faces)
        
        # Update attendance count (maximum faces seen)
        if current_face_count > self.attendance_count:
            self.attendance_count = current_face_count
            logger.info(f"Attendance updated: {self.attendance_count} people detected")
        
        # Update face positions for engagement analysis
        self.face_positions = [face['center'] for face in faces]
        
        # Keep face history for analysis (last 10 seconds)
        self.face_history.append({
            'timestamp': current_time,
            'face_count': current_face_count,
            'faces': faces
        })
        
        # Clean old history
        cutoff_time = current_time - 10.0  # 10 seconds
        self.face_history = [h for h in self.face_history if h['timestamp'] > cutoff_time]
    
    def get_attendance_stats(self) -> Dict[str, Any]:
        """Get attendance statistics"""
        if not self.face_history:
            return {
                'total_attendance': self.attendance_count,
                'current_present': 0,
                'average_present': 0.0,
                'attendance_trend': 'stable'
            }
        
        recent_counts = [h['face_count'] for h in self.face_history[-30:]]  # Last 30 frames
        current_present = recent_counts[-1] if recent_counts else 0
        average_present = sum(recent_counts) / len(recent_counts) if recent_counts else 0.0
        
        # Determine trend
        if len(recent_counts) >= 10:
            early_avg = sum(recent_counts[:5]) / 5
            late_avg = sum(recent_counts[-5:]) / 5
            if late_avg > early_avg * 1.1:
                trend = 'increasing'
            elif late_avg < early_avg * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'total_attendance': self.attendance_count,
            'current_present': current_present,
            'average_present': average_present,
            'attendance_trend': trend
        }
    
    def draw_detections(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw face detection results on frame"""
        result_frame = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Person: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = face['center']
            cv2.circle(result_frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        # Draw attendance info
        attendance_text = f"Attendance: {self.attendance_count} | Current: {len(faces)}"
        cv2.putText(result_frame, attendance_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
            self.model = None
        logger.info("Face detector cleaned up")
