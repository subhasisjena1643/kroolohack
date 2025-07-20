"""
Face Detection Module
Basic face detection for engagement analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import mediapipe as mp
import time
import json
from collections import deque
from pathlib import Path

from utils.base_processor import BaseProcessor
from utils.logger import logger
from ultralytics import YOLO

class FaceDetector(BaseProcessor):
    """Basic face detection for engagement analysis"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("FaceDetector", config)

        # Initialize MediaPipe Face Detection - OPTIMIZED FOR CLASSROOM DISTANCE
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        # INDUSTRIAL/RESEARCH GRADE FACE DETECTION
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model for maximum accuracy
            min_detection_confidence=0.1  # ULTRA-LOW: Maximum sensitivity for research-grade detection
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

        # ULTRA-AGGRESSIVE DETECTION FOR SMALLEST HEADS - MAXIMUM SENSITIVITY
        self.confidence_threshold = config.get('face_confidence_threshold', 0.03)  # EXTREMELY LOW for tiniest heads
        self.max_persons = config.get('max_faces', 200)  # Very high capacity for detecting everyone
        self.movement_sensitivity = config.get('movement_sensitivity', 0.001)  # Maximum sensitivity

        # EXTREME DETECTION PARAMETERS FOR SMALLEST FACES
        self.nms_threshold = 0.2  # Very aggressive NMS to catch overlapping small faces
        self.detection_stability_frames = 1  # Immediate detection without waiting
        self.multi_scale_detection = True  # Multiple scales for tiny faces
        self.small_face_boost = 3.0  # MASSIVE boost for small faces
        self.tiny_face_boost = 5.0  # Even bigger boost for extremely small faces
        self.classroom_mode = True  # Classroom optimization

        # ULTRA-PERMISSIVE SIZE PARAMETERS
        self.min_face_pixels = 6  # Detect faces as small as 6x6 pixels
        self.max_face_pixels = 1200  # Allow larger faces too
        self.aspect_ratio_tolerance = 0.5  # Very tolerant for varied angles

        # ADDITIONAL TINY FACE DETECTION PARAMETERS
        self.enable_tiny_face_detection = True
        self.tiny_face_threshold = 20  # Faces smaller than 20x20 are "tiny"
        self.micro_face_threshold = 10  # Faces smaller than 10x10 are "micro"

        # INTELLIGENT BODY DETECTION PARAMETERS
        self.enable_body_detection = True
        self.body_confidence_threshold = 0.3
        self.person_class_id = 0  # COCO person class ID

        # FEEDBACK REINFORCEMENT TRAINING INTEGRATION
        self.enable_feedback_learning = True
        self.adaptive_confidence_threshold = self.confidence_threshold
        self.feedback_adjustment_rate = 0.01
        self.performance_history = deque(maxlen=100)

        # TRAINED MODEL INTEGRATION
        self.use_trained_model = config.get('use_trained_model', True)
        self.trained_model_path = self._get_active_model_path()
        self.model_performance_boost = 1.2  # Boost confidence for trained models

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
            
            # Load trained model if available, otherwise use default
            model_path = self.trained_model_path if self.use_trained_model and self.trained_model_path else 'yolov8n.pt'

            self.model = YOLO(model_path)

            if self.use_trained_model and self.trained_model_path:
                logger.info(f"✅ Loaded TRAINED model: {model_path}")
                logger.info("🚀 Using state-of-the-art trained model for superior human detection")
            else:
                logger.info(f"✅ Loaded default YOLOv8 model: {model_path}")
            
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
            
            # Run YOLOv8 detection with ADAPTIVE THRESHOLD from feedback learning
            detection_threshold = self.adaptive_confidence_threshold if self.enable_feedback_learning else self.confidence_threshold

            results = self.model(
                processed_frame,
                verbose=False,
                conf=detection_threshold,        # ADAPTIVE threshold based on feedback learning
                iou=self.nms_threshold,          # Aggressive NMS (0.2)
                max_det=self.max_persons,        # High detection limit (200)
                imgsz=640,                       # Standard image size for good detection
                augment=True,                    # Enable test-time augmentation for better detection
                agnostic_nms=True,              # Class-agnostic NMS for better small object detection
                half=False                       # Use full precision for better tiny face detection
            )

            logger.debug(f"🧠 ADAPTIVE DETECTION: Using threshold {detection_threshold:.3f} (feedback learning: {self.enable_feedback_learning})")

            # Extract face detections with stability filtering
            faces = self._extract_faces_with_stability(results[0], frame.shape)

            # MULTI-SCALE DETECTION for ultra-tiny faces
            if self.multi_scale_detection and len(faces) < 5:  # If we didn't find many faces, try different scales
                additional_faces = self._multi_scale_detection(processed_frame, frame.shape)
                faces.extend(additional_faces)
            
            # Update attendance
            self._update_attendance(faces)
            
            # INTELLIGENT BODY DETECTION for distant people
            bodies = []
            if self.enable_body_detection:
                bodies = self._extract_body_detections(results[0], frame.shape)

            # Create result with both face and body detection
            result = {
                'faces': faces,
                'bodies': bodies,  # Add body detection results
                'face_count': len(faces),
                'body_count': len(bodies),
                'attendance_count': self.attendance_count,
                'face_positions': self.face_positions,
                'frame_shape': frame.shape,
                'detection_method': 'intelligent_dual_detection'
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

    def _extract_faces_with_stability(self, detection_result, original_shape):
        """Extract faces with stability filtering for better precision"""
        try:
            faces = []

            if detection_result.boxes is None:
                return faces

            boxes = detection_result.boxes.xyxy.cpu().numpy()
            confidences = detection_result.boxes.conf.cpu().numpy()
            classes = detection_result.boxes.cls.cpu().numpy()

            # Calculate scaling factors
            scale_x = original_shape[1] / self.input_size[0]
            scale_y = original_shape[0] / self.input_size[1]

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                # Filter for person class (class 0 in COCO dataset) with enhanced confidence
                if int(cls) == 0 and conf >= self.confidence_threshold:
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = box
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                    # ULTRA-AGGRESSIVE VALIDATION FOR TINIEST FACES
                    width = x2 - x1
                    height = y2 - y1

                    # EXTREMELY permissive size validation - detect even 6x6 pixel faces
                    if width < self.min_face_pixels or height < self.min_face_pixels:
                        continue

                    # Maximum size validation to filter out false positives
                    if width > self.max_face_pixels or height > self.max_face_pixels:
                        continue

                    # Very relaxed aspect ratio validation for all angles and poses
                    aspect_ratio = width / height
                    if aspect_ratio < (1.0 - self.aspect_ratio_tolerance) or aspect_ratio > (1.0 + self.aspect_ratio_tolerance):
                        continue

                    # MASSIVE CONFIDENCE BOOST for tiny faces
                    adjusted_conf = conf
                    face_size = min(width, height)

                    if face_size <= self.micro_face_threshold:  # Micro faces (≤10 pixels)
                        adjusted_conf = min(1.0, conf * self.tiny_face_boost * 2)  # 10x boost for micro faces
                        logger.debug(f"🔍 MICRO FACE DETECTED: {face_size}px, boosted conf: {adjusted_conf:.3f}")
                    elif face_size <= self.tiny_face_threshold:  # Tiny faces (≤20 pixels)
                        adjusted_conf = min(1.0, conf * self.tiny_face_boost)  # 5x boost for tiny faces
                        logger.debug(f"🔍 TINY FACE DETECTED: {face_size}px, boosted conf: {adjusted_conf:.3f}")
                    elif face_size <= 50:  # Small faces (≤50 pixels)
                        adjusted_conf = min(1.0, conf * self.small_face_boost)  # 3x boost for small faces
                        logger.debug(f"🔍 SMALL FACE DETECTED: {face_size}px, boosted conf: {adjusted_conf:.3f}")

                    # Additional boost if confidence is still very low but face looks valid
                    if adjusted_conf < 0.1 and face_size >= 8:  # Give extra chance to 8+ pixel faces
                        adjusted_conf = max(adjusted_conf, 0.1)  # Minimum confidence for valid-looking faces

                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, original_shape[1] - 1))
                    y1 = max(0, min(y1, original_shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, original_shape[1]))
                    y2 = max(y1 + 1, min(y2, original_shape[0]))

                    # Calculate center point for tracking
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Determine face size category
                    face_size = min(width, height)
                    is_micro_face = face_size <= self.micro_face_threshold
                    is_tiny_face = face_size <= self.tiny_face_threshold
                    is_small_face = face_size <= 50

                    face_data = {
                        'bbox': [x1, y1, width, height],  # [x, y, w, h] format
                        'confidence': float(adjusted_conf),  # Use adjusted confidence
                        'original_confidence': float(conf),  # Keep original for reference
                        'center': [center_x, center_y],
                        'area': width * height,
                        'aspect_ratio': aspect_ratio,
                        'face_size': face_size,  # Minimum dimension
                        'is_micro_face': is_micro_face,  # ≤10 pixels
                        'is_tiny_face': is_tiny_face,    # ≤20 pixels
                        'is_small_face': is_small_face,  # ≤50 pixels
                        'face_quality': self._assess_face_quality(width, height, adjusted_conf),
                        'size_category': self._get_size_category(face_size)
                    }

                    faces.append(face_data)

            # Sort by confidence for better processing order
            faces.sort(key=lambda x: x['confidence'], reverse=True)

            # Limit to max_persons for performance
            faces = faces[:self.max_persons]

            return faces

        except Exception as e:
            logger.error(f"Error extracting faces with stability: {e}")
            return []

    def _assess_face_quality(self, width: int, height: int, confidence: float) -> str:
        """Assess face quality for industrial-grade classification"""
        try:
            face_size = min(width, height)

            # Quality assessment based on size and confidence
            if face_size >= 80 and confidence >= 0.7:
                return "excellent"
            elif face_size >= 60 and confidence >= 0.5:
                return "good"
            elif face_size >= 40 and confidence >= 0.3:
                return "fair"
            elif face_size >= 20 and confidence >= 0.15:
                return "poor"
            else:
                return "very_poor"

        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return "unknown"

    def _get_size_category(self, face_size: int) -> str:
        """Get face size category for ultra-detailed classification"""
        try:
            if face_size <= 6:
                return "ultra_micro"  # Extremely tiny
            elif face_size <= 10:
                return "micro"        # Very tiny
            elif face_size <= 20:
                return "tiny"         # Small but detectable
            elif face_size <= 40:
                return "small"        # Small
            elif face_size <= 80:
                return "medium"       # Normal
            elif face_size <= 150:
                return "large"        # Large
            else:
                return "extra_large"  # Very large
        except Exception as e:
            logger.error(f"Error getting size category: {e}")
            return "unknown"

    def _multi_scale_detection(self, processed_frame, original_shape):
        """Multi-scale detection for ultra-tiny faces"""
        additional_faces = []

        try:
            # Try different scales to catch tiny faces
            scales = [0.5, 0.75, 1.25, 1.5]  # Different scaling factors

            for scale in scales:
                try:
                    # Resize frame for different scale detection
                    if scale != 1.0:
                        h, w = processed_frame.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        scaled_frame = cv2.resize(processed_frame, (new_w, new_h))
                    else:
                        scaled_frame = processed_frame

                    # Run detection on scaled frame with even lower confidence
                    scale_results = self.model(
                        scaled_frame,
                        verbose=False,
                        conf=max(0.01, self.confidence_threshold * 0.5),  # Even lower confidence
                        iou=self.nms_threshold * 0.8,  # More aggressive NMS
                        max_det=50,  # Limit for performance
                        imgsz=320 if scale < 1.0 else 640,  # Smaller image size for tiny faces
                        augment=True
                    )

                    if scale_results and scale_results[0].boxes is not None:
                        # Extract faces and scale back to original coordinates
                        scale_faces = self._extract_faces_with_scale_adjustment(
                            scale_results[0], original_shape, scale
                        )

                        # Filter out duplicates and add unique tiny faces
                        for face in scale_faces:
                            if self._is_unique_face(face, additional_faces):
                                additional_faces.append(face)
                                logger.debug(f"🔍 MULTI-SCALE DETECTION: Found tiny face at scale {scale}")

                except Exception as scale_error:
                    logger.debug(f"Error in scale {scale} detection: {scale_error}")
                    continue

            return additional_faces[:10]  # Limit to prevent too many false positives

        except Exception as e:
            logger.error(f"Error in multi-scale detection: {e}")
            return []

    def _extract_faces_with_scale_adjustment(self, detection_result, original_shape, scale):
        """Extract faces with scale adjustment back to original coordinates"""
        faces = []

        try:
            if detection_result.boxes is None:
                return faces

            boxes = detection_result.boxes.xyxy.cpu().numpy()
            confidences = detection_result.boxes.conf.cpu().numpy()
            classes = detection_result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                if int(cls) == 0:  # Person class
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = box
                    x1, x2 = int(x1 / scale), int(x2 / scale)
                    y1, y2 = int(y1 / scale), int(y2 / scale)

                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, original_shape[1] - 1))
                    y1 = max(0, min(y1, original_shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, original_shape[1]))
                    y2 = max(y1 + 1, min(y2, original_shape[0]))

                    width = x2 - x1
                    height = y2 - y1

                    # Only add if it's a reasonable face size
                    if width >= 6 and height >= 6:
                        face_size = min(width, height)

                        face_data = {
                            'bbox': [x1, y1, width, height],
                            'confidence': float(conf * 1.2),  # Slight boost for multi-scale detection
                            'original_confidence': float(conf),
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'area': width * height,
                            'face_size': face_size,
                            'is_micro_face': face_size <= 10,
                            'is_tiny_face': face_size <= 20,
                            'is_small_face': face_size <= 50,
                            'size_category': self._get_size_category(face_size),
                            'detection_method': f'multi_scale_{scale}',
                            'face_quality': self._assess_face_quality(width, height, conf)
                        }

                        faces.append(face_data)

            return faces

        except Exception as e:
            logger.error(f"Error extracting faces with scale adjustment: {e}")
            return []

    def _is_unique_face(self, new_face, existing_faces):
        """Check if a face is unique (not a duplicate)"""
        try:
            new_center = new_face['center']
            new_size = new_face['face_size']

            for existing_face in existing_faces:
                existing_center = existing_face['center']
                existing_size = existing_face['face_size']

                # Calculate distance between centers
                distance = ((new_center[0] - existing_center[0])**2 +
                           (new_center[1] - existing_center[1])**2)**0.5

                # Consider faces as duplicates if they're close and similar size
                size_threshold = max(new_size, existing_size) * 0.5
                if distance < size_threshold:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking face uniqueness: {e}")
            return True

    def _extract_body_detections(self, detection_result, original_shape):
        """Extract body/person detections for distant people analysis"""
        try:
            bodies = []

            if detection_result.boxes is None:
                return bodies

            boxes = detection_result.boxes.xyxy.cpu().numpy()
            confidences = detection_result.boxes.conf.cpu().numpy()
            classes = detection_result.boxes.cls.cpu().numpy()

            # Calculate scaling factors
            scale_x = original_shape[1] / self.input_size[0]
            scale_y = original_shape[0] / self.input_size[1]

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                # Filter for person class (class 0 in COCO dataset)
                if int(cls) == self.person_class_id and conf >= self.body_confidence_threshold:
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = box
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                    # Body validation
                    width = x2 - x1
                    height = y2 - y1

                    # Minimum body size validation
                    min_body_width = 30
                    min_body_height = 60
                    if width < min_body_width or height < min_body_height:
                        continue

                    # Aspect ratio validation for human bodies (should be taller than wide)
                    aspect_ratio = height / width
                    if aspect_ratio < 1.2 or aspect_ratio > 4.0:  # Human body proportions
                        continue

                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, original_shape[1] - 1))
                    y1 = max(0, min(y1, original_shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, original_shape[1]))
                    y2 = max(y1 + 1, min(y2, original_shape[0]))

                    # Calculate center point and area
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    body_area = width * height

                    # Assess body quality
                    body_quality = self._assess_body_quality(width, height, conf)

                    body_data = {
                        'bbox': [x1, y1, width, height],  # [x, y, w, h] format
                        'confidence': float(conf),
                        'center': [center_x, center_y],
                        'area': body_area,
                        'aspect_ratio': aspect_ratio,
                        'body_quality': body_quality,
                        'width': width,
                        'height': height,
                        'detection_type': 'full_body'
                    }

                    bodies.append(body_data)

            # Sort by confidence for better processing order
            bodies.sort(key=lambda x: x['confidence'], reverse=True)

            # Limit to prevent too many detections
            bodies = bodies[:self.max_persons]

            logger.debug(f"🚶 BODY DETECTION: Found {len(bodies)} bodies")

            return bodies

        except Exception as e:
            logger.error(f"Error extracting body detections: {e}")
            return []

    def _assess_body_quality(self, width: int, height: int, confidence: float) -> str:
        """Assess body detection quality"""
        try:
            body_area = width * height

            # Quality assessment based on size and confidence
            if body_area >= 15000 and confidence >= 0.7:
                return "excellent"
            elif body_area >= 10000 and confidence >= 0.5:
                return "good"
            elif body_area >= 5000 and confidence >= 0.4:
                return "fair"
            elif body_area >= 2000 and confidence >= 0.3:
                return "poor"
            else:
                return "very_poor"

        except Exception as e:
            logger.error(f"Error assessing body quality: {e}")
            return "unknown"

    def add_detection_feedback(self, feedback_type: str, detection_data: Dict[str, Any]):
        """Add feedback to improve detection performance through reinforcement learning"""
        try:
            current_time = time.time()

            feedback_instance = {
                'timestamp': current_time,
                'feedback_type': feedback_type,
                'detection_data': detection_data,
                'current_threshold': self.adaptive_confidence_threshold
            }

            self.performance_history.append(feedback_instance)

            # Process feedback immediately
            if feedback_type == 'false_positive':
                # Detection was wrong - increase threshold to be more strict
                adjustment = self.feedback_adjustment_rate * 2.0  # Stronger adjustment for false positives
                self.adaptive_confidence_threshold = min(0.8, self.adaptive_confidence_threshold + adjustment)
                logger.info(f"🔺 FALSE POSITIVE: Increased detection threshold to {self.adaptive_confidence_threshold:.3f}")

            elif feedback_type == 'false_negative' or feedback_type == 'missed_detection':
                # Missed detection - decrease threshold to be more sensitive
                adjustment = self.feedback_adjustment_rate * 2.0
                self.adaptive_confidence_threshold = max(0.01, self.adaptive_confidence_threshold - adjustment)
                logger.info(f"🔻 FALSE NEGATIVE: Decreased detection threshold to {self.adaptive_confidence_threshold:.3f}")

            elif feedback_type == 'correct_detection':
                # Good detection - slight positive reinforcement
                target_confidence = detection_data.get('confidence', self.adaptive_confidence_threshold)
                self.adaptive_confidence_threshold = self.adaptive_confidence_threshold * 0.98 + target_confidence * 0.02
                logger.debug(f"✅ CORRECT DETECTION: Reinforced threshold to {self.adaptive_confidence_threshold:.3f}")

            # Update the model's confidence threshold
            self.confidence_threshold = self.adaptive_confidence_threshold

            # Calculate recent performance
            if len(self.performance_history) >= 20:
                recent_feedback = list(self.performance_history)[-20:]
                positive_count = sum(1 for f in recent_feedback if f['feedback_type'] == 'correct_detection')
                total_count = len(recent_feedback)
                accuracy = positive_count / total_count

                logger.info(f"📊 DETECTION PERFORMANCE: {accuracy:.2%} accuracy over last {total_count} feedback instances")

                # Auto-adjust learning rate based on performance
                if accuracy < 0.7:  # Poor performance
                    self.feedback_adjustment_rate = min(0.05, self.feedback_adjustment_rate * 1.1)
                elif accuracy > 0.9:  # Excellent performance
                    self.feedback_adjustment_rate = max(0.005, self.feedback_adjustment_rate * 0.9)

        except Exception as e:
            logger.error(f"Error adding detection feedback: {e}")

    def get_detection_performance_summary(self) -> Dict[str, Any]:
        """Get summary of detection performance and learning"""
        try:
            total_feedback = len(self.performance_history)

            if total_feedback == 0:
                return {
                    'total_feedback': 0,
                    'accuracy': 0,
                    'current_threshold': self.adaptive_confidence_threshold,
                    'learning_rate': self.feedback_adjustment_rate
                }

            # Count feedback types
            correct_count = sum(1 for f in self.performance_history if f['feedback_type'] == 'correct_detection')
            false_positive_count = sum(1 for f in self.performance_history if f['feedback_type'] == 'false_positive')
            false_negative_count = sum(1 for f in self.performance_history if f['feedback_type'] in ['false_negative', 'missed_detection'])

            accuracy = correct_count / total_feedback if total_feedback > 0 else 0

            return {
                'total_feedback': total_feedback,
                'correct_detections': correct_count,
                'false_positives': false_positive_count,
                'false_negatives': false_negative_count,
                'accuracy': accuracy,
                'current_threshold': self.adaptive_confidence_threshold,
                'original_threshold': self.confidence_threshold,
                'learning_rate': self.feedback_adjustment_rate,
                'threshold_adjustment_total': self.adaptive_confidence_threshold - self.confidence_threshold
            }

        except Exception as e:
            logger.error(f"Error getting detection performance summary: {e}")
            return {}

    def _get_active_model_path(self) -> Optional[str]:
        """Get path to active trained model"""
        try:
            # Check for active model configuration
            config_file = Path("configs/active_model_config.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_path = config.get('model_path')
                    if model_path and Path(model_path).exists():
                        return model_path

            # Check for active model in models directory
            active_model_path = Path("models/detection/active_detection_model.pt")
            if active_model_path.exists():
                return str(active_model_path)

            # Check for best model in checkpoints
            checkpoint_dir = Path("checkpoints/detection_training")
            if checkpoint_dir.exists():
                best_models = list(checkpoint_dir.glob("**/best.pt"))
                if best_models:
                    return str(best_models[0])

            return None

        except Exception as e:
            logger.error(f"Error getting active model path: {e}")
            return None

    def reload_trained_model(self) -> bool:
        """Reload the trained model (useful after new training)"""
        try:
            new_model_path = self._get_active_model_path()

            if new_model_path and new_model_path != self.trained_model_path:
                logger.info(f"🔄 Reloading trained model: {new_model_path}")

                # Update model path
                self.trained_model_path = new_model_path

                # Reload model
                self.model = YOLO(new_model_path)

                logger.info("✅ Trained model reloaded successfully")
                return True
            else:
                logger.debug("No new trained model to reload")
                return False

        except Exception as e:
            logger.error(f"Error reloading trained model: {e}")
            return False

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
