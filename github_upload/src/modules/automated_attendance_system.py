"""
Automated Attendance System with Facial Recognition
Real-time face recognition, tracking, and attendance logging
"""

import cv2
import numpy as np
import sqlite3
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully - Advanced facial recognition enabled")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not available. Running in detection-only mode.")
import json
import time
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import logging
import shutil

from src.utils.base_processor import BaseProcessor
from src.utils.logger import logger

# Try to import continuous learning system
try:
    from .continuous_learning_system import ContinuousLearningSystem, FeedbackType, LearningInstance
    CONTINUOUS_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUOUS_LEARNING_AVAILABLE = False
    logger.warning("Continuous Learning System not available - running without advanced learning features")

@dataclass
class Student:
    """Student information structure"""
    roll_number: str
    name: str
    application_number: str
    photo_path: str
    face_encoding: np.ndarray
    department: str = ""
    year: str = ""
    section: str = ""

@dataclass
class AttendanceRecord:
    """Attendance record structure"""
    roll_number: str
    name: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    total_duration: float = 0.0
    engagement_score: float = 0.0
    participation_score: float = 0.0
    attention_score: float = 0.0

@dataclass
class TrackedPerson:
    """Tracked person in live video"""
    person_id: str
    roll_number: str
    name: str
    face_bbox: List[int]
    face_encoding: np.ndarray
    last_seen: float
    tracking_confidence: float
    entry_time: datetime
    confidence: float = 0.0  # Face recognition confidence
    is_present: bool = True
    alert_triggered: bool = False
    alert_start_time: Optional[float] = None

class AutomatedAttendanceSystem(BaseProcessor):
    """Industrial-grade automated attendance system with facial recognition"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AutomatedAttendanceSystem", config)
        
        # Configuration
        self.dataset_path = config.get('student_dataset_path', 'data/student_dataset')
        self.database_path = config.get('attendance_db_path', 'data/attendance.db')
        # INDUSTRIAL/RESEARCH GRADE DETECTION PARAMETERS - ENHANCED FOR REAL-WORLD CLASSROOMS
        # Optimized for detecting ALL people (known + unknown) with high precision
        self.base_face_recognition_threshold = config.get('face_recognition_threshold', 0.45)  # Optimized for classroom diversity
        self.face_recognition_threshold = self.base_face_recognition_threshold
        self.dynamic_threshold = self.base_face_recognition_threshold

        # ENHANCED DETECTION PARAMETERS FOR ALL PEOPLE (KNOWN + UNKNOWN)
        self.face_detection_confidence = config.get('face_detection_confidence', 0.12)  # Ultra-low for maximum detection
        self.min_face_size = config.get('min_face_size', 15)  # Detect even very small/distant faces
        self.max_face_distance = config.get('max_face_distance', 15.0)  # Extended range for classroom scenarios
        self.unknown_person_tracking = True  # Track unknown people too
        self.multi_scale_detection = True  # Detect faces at multiple scales
        self.adaptive_threshold_enabled = True  # Dynamic threshold adjustment
        self.small_face_enhancement = True  # Special processing for small faces
        self.classroom_optimization = True  # Optimized for classroom environments

        # RELAXED CONFIRMATION SYSTEM (for better continuous detection)
        self.recognition_confirmation_frames = 2  # Reduced to 2 frames for faster confirmation
        self.recognition_buffer = {}  # person_id -> [recent_recognitions]
        self.max_buffer_size = 5

        # RELAXED VALIDATION SYSTEM (for better continuous detection)
        self.min_face_area = 5000  # Much more lenient minimum face area
        self.max_recognition_distance = 100  # More lenient pixel distance
        self.confidence_threshold = 0.5  # More lenient face detection confidence

        # RELAXED TEMPORAL CONSISTENCY CHECKS
        self.temporal_window = 15  # More frames to check for consistency
        self.consistency_threshold = 0.6  # 60% of frames must agree on identity

        # REINFORCEMENT LEARNING SYSTEM
        self.reinforcement_learning_enabled = True
        self.learning_rate = 0.01
        self.experience_buffer = []  # Store learning experiences
        self.max_experience_buffer_size = 1000

        # ADAPTIVE THRESHOLD LEARNING
        self.threshold_history = []
        self.recognition_success_history = []
        self.false_positive_history = []

        # FACE DETECTION IMPROVEMENT
        self.detection_confidence_history = []
        self.detection_quality_scores = []
        self.optimal_detection_params = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'scale_factor': 1.1
        }

        # RECOGNITION ACCURACY TRACKING
        self.recognition_accuracy_tracker = {
            'correct_recognitions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_attempts': 0
        }

        # CONTINUOUS LEARNING METRICS
        self.learning_metrics = {
            'sessions_count': 0,
            'total_faces_processed': 0,
            'recognition_improvement_rate': 0.0,
            'detection_improvement_rate': 0.0,
            'last_model_update': time.time()
        }
        self.tracking_threshold = config.get('tracking_threshold', 0.7)
        self.disappearance_alert_duration = config.get('disappearance_alert_duration', 30.0)  # 30 seconds
        self.max_tracking_distance = config.get('max_tracking_distance', 100)  # pixels
        
        # Student database
        self.students_db: Dict[str, Student] = {}
        self.known_face_encodings: List[np.ndarray] = []
        self.known_roll_numbers: List[str] = []
        
        # ENHANCED TRACKING FOR ALL PEOPLE (KNOWN + UNKNOWN)
        self.tracked_persons: Dict[str, TrackedPerson] = {}  # Known people
        self.unknown_persons: Dict[str, TrackedPerson] = {}  # Unknown people tracking
        self.next_person_id = 1
        self.next_unknown_id = 1
        self.attendance_records: Dict[str, AttendanceRecord] = {}

        # COMPREHENSIVE TRACKING PARAMETERS
        self.track_unknown_people = True
        self.unknown_person_engagement_tracking = True
        self.calculate_unknown_person_scores = True
        self.unknown_person_parameters = {}  # Store calculated parameters for unknown people

        # INTELLIGENT DUAL-DETECTION SYSTEM
        self.enable_intelligent_detection = True
        self.body_detection_enabled = True
        self.face_body_fusion_enabled = True
        self.distance_based_analysis = True

        # DETECTION THRESHOLDS FOR INTELLIGENT SWITCHING
        self.face_size_threshold_for_body = 30  # If face < 30px, prioritize body detection
        self.body_confidence_threshold = 0.3   # Minimum body detection confidence
        self.fusion_distance_threshold = 100   # Distance for face-body association

        # FEEDBACK REINFORCEMENT TRAINING SYSTEM
        self.enable_feedback_training = True
        self.feedback_buffer = deque(maxlen=1000)  # Store feedback instances
        self.negative_feedback_buffer = deque(maxlen=500)  # Focus on negative feedback
        self.positive_feedback_buffer = deque(maxlen=500)  # Track positive feedback
        self.detection_performance_history = deque(maxlen=200)  # Performance tracking

        # ADAPTIVE THRESHOLDS BASED ON FEEDBACK
        self.adaptive_face_threshold = self.face_detection_confidence
        self.adaptive_body_threshold = self.body_confidence_threshold
        self.adaptive_recognition_threshold = self.face_recognition_threshold

        # REINFORCEMENT LEARNING PARAMETERS
        self.learning_rate_detection = 0.01
        self.feedback_weight_negative = 2.0  # Weight negative feedback more heavily
        self.feedback_weight_positive = 1.0
        self.threshold_adjustment_factor = 0.02
        self.performance_window_size = 50
        
        # Performance optimization (balanced for speed and accuracy)
        self.face_recognition_interval = config.get('face_recognition_interval', 3)  # Every 3 frames for balanced performance
        self.frame_count = 0

        # INDUSTRIAL-GRADE BODY TRACKING
        self.tracked_bodies: Dict[str, Dict] = {}  # person_id -> tracker_info
        self.face_recognition_mode = True  # Start with face recognition, switch to body tracking
        self.body_tracking_confidence_threshold = 0.2  # More lenient threshold

        # Multi-tracker ensemble for robustness
        self.available_trackers = ['CSRT', 'KCF', 'MOSSE']  # Multiple tracking algorithms
        self.tracker_weights = {'CSRT': 0.5, 'KCF': 0.3, 'MOSSE': 0.2}  # Weighted ensemble

        # IMMEDIATE FACE LOCKING SYSTEM
        self.locked_faces: Dict[str, Dict] = {}  # person_id -> locked_face_info
        self.face_lock_timeout = 30.0  # Keep locked face for 30 seconds without detection
        self.immediate_lock_enabled = True  # Enable immediate locking on detection

        # Alerts and notifications
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[callable] = []

        # Latest result cache for display
        self.latest_result = None

        # Database connection
        self.db_connection = None

        # CONTINUOUS LEARNING INTEGRATION WITH CHECKPOINT PERSISTENCE
        self.continuous_learning_enabled = config.get('continuous_learning_enabled', True)
        self.learning_buffer = deque(maxlen=1000)  # Store recent learning instances
        self.recognition_history = deque(maxlen=500)  # Track recognition accuracy over time
        self.face_detection_history = deque(maxlen=500)  # Track detection performance
        self.confidence_threshold_history = deque(maxlen=100)  # Track threshold adjustments

        # Reinforcement Learning Parameters (will be loaded from checkpoint)
        self.learning_rate = 0.01
        self.exploration_rate = 0.1  # For epsilon-greedy exploration
        self.reward_decay = 0.95
        self.confidence_adjustment_factor = 0.02

        # Performance Tracking (will be loaded from checkpoint)
        self.session_start_time = time.time()
        self.total_detections = 0
        self.correct_recognitions = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.detection_accuracy_window = deque(maxlen=50)  # Rolling accuracy window

        # CHECKPOINT PERSISTENCE SYSTEM
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'attendance_system')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'learning_checkpoint.pkl')
        self.auto_save_interval = 100  # Save checkpoint every 100 detections
        self.last_checkpoint_save = 0

        # LOAD CHECKPOINT BEFORE INITIALIZING LEARNING SYSTEM
        self._load_learning_checkpoint()

        # External learning system will be set by the main application
        self.external_learning_system = None
        logger.info("🧠 Attendance system ready for external continuous learning integration")

    def set_external_learning_system(self, learning_system):
        """Set the external continuous learning system"""
        self.external_learning_system = learning_system
        logger.info("🧠 External continuous learning system connected to attendance system")
        
    def initialize(self) -> bool:
        """Initialize the automated attendance system"""
        try:
            logger.info("Initializing Automated Attendance System...")
            
            # Initialize database
            if not self._initialize_database():
                return False
            
            # Load student dataset
            if not self._load_student_dataset():
                logger.warning("No student dataset found. System will work in detection-only mode.")
            
            # Create necessary directories
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            os.makedirs(self.dataset_path, exist_ok=True)
            
            logger.info(f"Loaded {len(self.students_db)} students from dataset")
            logger.info("Automated Attendance System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize attendance system: {e}")
            return False
    
    def _initialize_database(self) -> bool:
        """Initialize SQLite database for attendance records"""
        try:
            self.db_connection = sqlite3.connect(self.database_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roll_number TEXT NOT NULL,
                    name TEXT NOT NULL,
                    date DATE NOT NULL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    total_duration REAL,
                    engagement_score REAL,
                    participation_score REAL,
                    attention_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    roll_number TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    application_number TEXT UNIQUE,
                    department TEXT,
                    year TEXT,
                    section TEXT,
                    photo_path TEXT,
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roll_number TEXT,
                    alert_type TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _load_student_dataset(self) -> bool:
        """Load student dataset with photos and face encodings"""
        try:
            dataset_file = os.path.join(self.dataset_path, 'students.json')
            encodings_file = os.path.join(self.dataset_path, 'face_encodings.pkl')
            
            # Load student information
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r') as f:
                    students_data = json.load(f)
                
                # Always regenerate face encodings to ensure compatibility with current model
                logger.info("Regenerating face encodings for model compatibility...")
                encodings_data = self._generate_face_encodings(students_data)
                with open(encodings_file, 'wb') as f:
                    pickle.dump(encodings_data, f)
                
                # Build student database
                for student_data in students_data:
                    roll_number = student_data['roll_number']
                    if roll_number in encodings_data:
                        student = Student(
                            roll_number=roll_number,
                            name=student_data['name'],
                            application_number=student_data.get('application_number', ''),
                            photo_path=student_data.get('photo_path', ''),
                            face_encoding=encodings_data[roll_number],
                            department=student_data.get('department', ''),
                            year=student_data.get('year', ''),
                            section=student_data.get('section', '')
                        )
                        self.students_db[roll_number] = student
                        self.known_face_encodings.append(student.face_encoding)
                        self.known_roll_numbers.append(roll_number)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load student dataset: {e}")
            return False
    
    def _generate_face_encodings(self, students_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate face encodings from student photos using DeepFace"""
        encodings = {}

        if not DEEPFACE_AVAILABLE:
            logger.warning("DeepFace not available. Skipping encoding generation.")
            return encodings

        logger.info("Generating face encodings using DeepFace...")

        for student_data in students_data:
            try:
                photo_path = student_data.get('photo_path', '')
                if photo_path and os.path.exists(photo_path):
                    # Check if file is not empty
                    if os.path.getsize(photo_path) == 0:
                        logger.warning(f"Empty photo file for {student_data['name']}")
                        continue

                    # Generate embedding using DeepFace
                    try:
                        embedding_objs = DeepFace.represent(
                            img_path=photo_path,
                            model_name='Facenet',  # High accuracy model
                            enforce_detection=True,
                            detector_backend='opencv'
                        )

                        if embedding_objs and len(embedding_objs) > 0:
                            # Get the first face embedding
                            embedding = np.array(embedding_objs[0]['embedding'])
                            encodings[student_data['roll_number']] = embedding
                            logger.info(f"✅ Generated DeepFace embedding for {student_data['name']} ({student_data['roll_number']})")
                        else:
                            logger.warning(f"No face detected in photo for {student_data['name']}")

                    except Exception as deepface_error:
                        logger.warning(f"DeepFace failed for {student_data['name']}: {deepface_error}")

            except Exception as e:
                logger.error(f"Failed to process photo for {student_data.get('name', 'Unknown')}: {e}")

        logger.info(f"Generated {len(encodings)} face encodings using DeepFace")
        return encodings

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """INTELLIGENT DUAL-DETECTION: Process frame data with intelligent face/body detection switching"""
        try:
            frame = data.get('frame')
            face_data = data.get('face_detection', {})

            if frame is None:
                return self._empty_result()

            self.frame_count += 1
            current_time = time.time()

            # Store current faces for annotation display (for continuous face box detection)
            faces = face_data.get('faces', [])
            self.current_faces = faces

            # INTELLIGENT DETECTION COORDINATION
            detection_results = self._intelligent_detection_coordinator(data, frame, current_time)
            recognition_results = detection_results.get('recognition_results', [])

            # Update locked faces with current detections
            self._update_locked_faces(current_time, recognition_results)

            # Update tracking for all persons (intelligent mode)
            self._update_tracking(faces, recognition_results, current_time)

            # Check for disappearances and generate alerts
            self._check_disappearances(current_time)

            # Update attendance records
            self._update_attendance_records(current_time)

            # Prepare result
            result = {
                'tracked_persons': self._get_tracked_persons_summary(),
                'attendance_count': len([p for p in self.tracked_persons.values() if p.is_present]),
                'total_recognized': len(self.attendance_records),
                'active_alerts': list(self.active_alerts.values()),
                'frame_annotations': self._get_frame_annotations()
            }

            # Cache the latest result for display
            self.latest_result = result

            return result

        except Exception as e:
            logger.error(f"Error in attendance processing: {e}")
            return self._empty_result()

    def _initialize_body_tracking(self, frame: np.ndarray, recognition_results: List[Dict]):
        """Initialize body tracking for recognized persons"""
        try:
            for result in recognition_results:
                person_id = f"person_{result['roll_number']}"

                # Skip if already tracking this person
                if person_id in self.tracked_bodies:
                    logger.debug(f"Already tracking {result['name']}, skipping initialization")
                    continue

                bbox = result['face_bbox']

                # Expand face bbox to approximate body region
                x1, y1, x2, y2 = bbox
                face_width = x2 - x1
                face_height = y2 - y1

                # Estimate body bbox (more conservative)
                body_x1 = max(0, x1 - face_width // 3)
                body_y1 = max(0, y1 - face_height // 6)
                body_x2 = min(frame.shape[1], x2 + face_width // 3)
                body_y2 = min(frame.shape[0], y1 + face_height * 3)  # Body is ~3x face height

                body_width = body_x2 - body_x1
                body_height = body_y2 - body_y1

                # Ensure minimum size
                if body_width < 50 or body_height < 100:
                    logger.warning(f"Body bbox too small for {result['name']}, skipping body tracking")
                    continue

                body_bbox = (body_x1, body_y1, body_width, body_height)

                # INDUSTRIAL-GRADE MULTI-TRACKER INITIALIZATION
                trackers = {}
                successful_trackers = 0

                for tracker_type in self.available_trackers:
                    try:
                        if tracker_type == 'CSRT' and hasattr(cv2, 'TrackerCSRT_create'):
                            tracker = cv2.TrackerCSRT_create()
                        elif tracker_type == 'KCF' and hasattr(cv2, 'TrackerKCF_create'):
                            tracker = cv2.TrackerKCF_create()
                        elif tracker_type == 'MOSSE' and hasattr(cv2, 'TrackerMOSSE_create'):
                            tracker = cv2.TrackerMOSSE_create()
                        else:
                            continue

                        success = tracker.init(frame, body_bbox)
                        if success:
                            trackers[tracker_type] = {
                                'tracker': tracker,
                                'confidence': 1.0,
                                'weight': self.tracker_weights.get(tracker_type, 0.33)
                            }
                            successful_trackers += 1
                            logger.debug(f"✅ {tracker_type} tracker initialized for {result['name']}")
                        else:
                            logger.debug(f"❌ {tracker_type} tracker failed to initialize")

                    except Exception as tracker_error:
                        logger.debug(f"❌ {tracker_type} tracker error: {tracker_error}")
                        continue

                if successful_trackers > 0:
                    self.tracked_bodies[person_id] = {
                        'trackers': trackers,
                        'bbox': body_bbox,
                        'roll_number': result['roll_number'],
                        'name': result['name'],
                        'confidence': result['confidence'],
                        'last_update': time.time(),
                        'tracking_confidence': 1.0,
                        'init_frame': self.frame_count,
                        'successful_trackers': successful_trackers
                    }
                    logger.info(f"🎯 INDUSTRIAL BODY TRACKING LOCKED: {result['name']} ({result['roll_number']}) - {successful_trackers} trackers")
                else:
                    logger.warning(f"❌ All trackers failed for {result['name']}")

            # Switch to body tracking mode if we have tracked bodies
            if self.tracked_bodies:
                self.face_recognition_mode = False
                logger.info(f"🔒 LOCKED to body tracking mode for {len(self.tracked_bodies)} persons")

        except Exception as e:
            logger.error(f"Error initializing body tracking: {e}")

    def _update_body_tracking(self, frame: np.ndarray, current_time: float) -> List[Dict]:
        """Update industrial-grade multi-tracker body tracking"""
        recognition_results = []
        bodies_to_remove = []

        try:
            for person_id, body_info in self.tracked_bodies.items():
                trackers = body_info['trackers']

                # ENSEMBLE TRACKING: Update all trackers and combine results
                successful_updates = 0
                weighted_bbox = None
                tracker_results = []

                for tracker_type, tracker_data in trackers.items():
                    try:
                        success, bbox = tracker_data['tracker'].update(frame)

                        if success and bbox[2] > 10 and bbox[3] > 10:  # Valid bbox
                            tracker_results.append({
                                'type': tracker_type,
                                'bbox': bbox,
                                'weight': tracker_data['weight'],
                                'confidence': tracker_data['confidence']
                            })
                            successful_updates += 1
                            # Increase confidence for successful tracker
                            tracker_data['confidence'] = min(1.0, tracker_data['confidence'] + 0.05)
                        else:
                            # Decrease confidence for failed tracker
                            tracker_data['confidence'] = max(0.0, tracker_data['confidence'] - 0.1)

                    except Exception as tracker_error:
                        logger.debug(f"Tracker {tracker_type} failed: {tracker_error}")
                        tracker_data['confidence'] = max(0.0, tracker_data['confidence'] - 0.2)

                # WEIGHTED ENSEMBLE: Combine successful tracker results
                if successful_updates > 0:
                    # Calculate weighted average of bounding boxes
                    total_weight = sum(result['weight'] * result['confidence'] for result in tracker_results)

                    if total_weight > 0:
                        weighted_x = sum(result['bbox'][0] * result['weight'] * result['confidence'] for result in tracker_results) / total_weight
                        weighted_y = sum(result['bbox'][1] * result['weight'] * result['confidence'] for result in tracker_results) / total_weight
                        weighted_w = sum(result['bbox'][2] * result['weight'] * result['confidence'] for result in tracker_results) / total_weight
                        weighted_h = sum(result['bbox'][3] * result['weight'] * result['confidence'] for result in tracker_results) / total_weight

                        weighted_bbox = (weighted_x, weighted_y, weighted_w, weighted_h)

                        # Update body info with ensemble result
                        body_info['bbox'] = weighted_bbox
                        body_info['last_update'] = current_time
                        body_info['tracking_confidence'] = min(1.0, body_info['tracking_confidence'] + 0.03)

                        # Convert to recognition result format
                        x, y, w, h = [int(v) for v in weighted_bbox]
                        recognition_results.append({
                            'face_bbox': [x, y, x + w, y + h],
                            'roll_number': body_info['roll_number'],
                            'name': body_info['name'],
                            'confidence': body_info['confidence'],
                            'tracking_confidence': body_info['tracking_confidence'],
                            'tracking_method': 'industrial_ensemble',
                            'is_body_tracking': True,
                            'active_trackers': successful_updates
                        })

                        logger.debug(f"🎯 ENSEMBLE TRACKING: {body_info['name']} - conf: {body_info['tracking_confidence']:.2f} ({successful_updates}/{len(trackers)} trackers)")
                    else:
                        successful_updates = 0  # No valid weighted result

                # Handle tracking failures
                if successful_updates == 0:
                    body_info['tracking_confidence'] = max(0.0, body_info['tracking_confidence'] - 0.15)

                    # Only remove after confidence drops very low
                    if body_info['tracking_confidence'] < self.body_tracking_confidence_threshold:
                        bodies_to_remove.append(person_id)
                        logger.warning(f"❌ ENSEMBLE TRACKING LOST: {body_info['name']} ({body_info['roll_number']})")
                    else:
                        logger.debug(f"⚠️ Ensemble tracking struggling: {body_info['name']} - conf: {body_info['tracking_confidence']:.2f}")

            # Remove failed trackers
            for person_id in bodies_to_remove:
                del self.tracked_bodies[person_id]
                logger.info(f"🗑️ Removed ensemble tracker for {person_id}")

            # Switch back to face recognition if all bodies lost
            if not self.tracked_bodies and not self.face_recognition_mode:
                self.face_recognition_mode = True
                logger.info("🔄 Switched back to face recognition mode - all ensemble trackers lost")

        except Exception as e:
            logger.error(f"Error in ensemble body tracking: {e}")

        return recognition_results

    def _update_tracking_from_bodies(self, recognition_results: List[Dict], current_time: float):
        """Update person tracking based on body tracking results"""
        try:
            # Mark all persons as not seen initially
            for person in self.tracked_persons.values():
                person.seen_this_frame = False

            # Process body tracking results
            for result in recognition_results:
                roll_number = result['roll_number']
                bbox = result['face_bbox']  # Actually body bbox in this case
                center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

                # Find or create tracked person
                person = self._find_or_create_person(roll_number, center, current_time)

                if person:
                    # Update person with body tracking info
                    person.update_position(center, current_time)
                    person.roll_number = roll_number
                    person.name = result['name']
                    person.confidence = result['confidence']
                    person.tracking_confidence = result.get('tracking_confidence', 1.0)
                    person.is_present = True
                    person.seen_this_frame = True
                    person.alert_triggered = False  # Reset alert when person is seen

                    # Log attendance
                    self._log_attendance(roll_number, 'present', current_time)

                    logger.debug(f"👤 Body tracking: {person.name} ({person.roll_number}) at {center}")

            # Handle persons not seen this frame
            for person in self.tracked_persons.values():
                if not person.seen_this_frame and person.is_present:
                    person.is_present = False
                    person.last_seen = current_time
                    logger.debug(f"👻 Person not seen: {person.name} ({person.roll_number})")

        except Exception as e:
            logger.error(f"Error updating tracking from bodies: {e}")

    def _preprocess_face_for_recognition(self, face_region: np.ndarray) -> np.ndarray:
        """Industrial-grade face preprocessing for robust recognition"""
        try:
            # Ensure minimum size
            if face_region.shape[0] < 50 or face_region.shape[1] < 50:
                face_region = cv2.resize(face_region, (112, 112))

            # Convert to RGB if needed
            if len(face_region.shape) == 3 and face_region.shape[2] == 3:
                face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

            # Lighting normalization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if self.config.get('lighting_normalization', True):
                if len(face_region.shape) == 3:
                    # Convert to LAB color space for better lighting normalization
                    lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply to L channel
                    face_region = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    face_region = clahe.apply(face_region)

            # Gaussian blur to reduce noise
            face_region = cv2.GaussianBlur(face_region, (3, 3), 0)

            # Ensure proper data type
            face_region = face_region.astype(np.uint8)

            logger.debug(f"Face preprocessed: {face_region.shape}, dtype: {face_region.dtype}")
            return face_region

        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return face_region  # Return original if preprocessing fails

    def _calculate_adaptive_threshold(self, face_data: Dict, similarity_score: float) -> float:
        """Calculate adaptive threshold based on face conditions for real-world robustness"""
        try:
            base_threshold = self.base_face_recognition_threshold

            # Start with base threshold
            adaptive_threshold = base_threshold

            # Analyze face detection confidence
            face_confidence = face_data.get('confidence', 0.5)

            # Lower threshold for high-confidence face detections
            if face_confidence > 0.9:
                adaptive_threshold -= 0.05  # More lenient for clear faces
            elif face_confidence < 0.7:
                adaptive_threshold -= 0.03  # Still more lenient for unclear faces

            # Analyze face size (larger faces are typically clearer)
            bbox = face_data.get('bbox', [0, 0, 100, 100])
            if len(bbox) >= 4:
                face_area = bbox[2] * bbox[3]  # width * height
                if face_area > 20000:  # Large face
                    adaptive_threshold -= 0.03
                elif face_area < 8000:  # Small face
                    adaptive_threshold -= 0.05  # More lenient for small faces

            # Real-world condition adaptations
            # These would ideally be detected from the image, but we'll use conservative adjustments

            # General real-world tolerance (lighting, pose, accessories)
            adaptive_threshold -= self.lighting_tolerance
            adaptive_threshold -= self.pose_tolerance
            adaptive_threshold -= self.accessory_tolerance

            # Ensure threshold doesn't go too low
            min_threshold = 0.25  # Absolute minimum to prevent false positives
            adaptive_threshold = max(adaptive_threshold, min_threshold)

            # Ensure threshold doesn't go too high
            max_threshold = 0.6
            adaptive_threshold = min(adaptive_threshold, max_threshold)

            logger.debug(f"Adaptive threshold: {adaptive_threshold:.3f} (base: {base_threshold:.3f}, face_conf: {face_confidence:.3f})")

            return adaptive_threshold

        except Exception as e:
            logger.error(f"Error calculating adaptive threshold: {e}")
            return self.base_face_recognition_threshold

    def _confirm_recognition_multi_frame(self, roll_number: str, confidence: float, bbox: List[int], current_time: float) -> bool:
        """Multi-frame confirmation system to prevent false positives"""
        try:
            # Initialize buffer for this person if not exists
            if roll_number not in self.recognition_buffer:
                self.recognition_buffer[roll_number] = []

            # Add current recognition to buffer
            recognition_data = {
                'confidence': confidence,
                'bbox': bbox,
                'timestamp': current_time
            }

            self.recognition_buffer[roll_number].append(recognition_data)

            # Keep only recent recognitions
            if len(self.recognition_buffer[roll_number]) > self.max_buffer_size:
                self.recognition_buffer[roll_number].pop(0)

            # Check if we have enough consecutive confirmations
            recent_recognitions = self.recognition_buffer[roll_number]

            if len(recent_recognitions) >= self.recognition_confirmation_frames:
                # Check temporal consistency (recognitions should be close in time)
                time_span = recent_recognitions[-1]['timestamp'] - recent_recognitions[-self.recognition_confirmation_frames]['timestamp']

                if time_span < 15.0:  # Within 15 seconds (much more lenient)
                    # Check spatial consistency (face positions should be close)
                    spatial_consistent = self._check_spatial_consistency(recent_recognitions[-self.recognition_confirmation_frames:])

                    if spatial_consistent:
                        # Check confidence consistency (all should be above threshold)
                        confidence_consistent = all(r['confidence'] > self.face_recognition_threshold for r in recent_recognitions[-self.recognition_confirmation_frames:])

                        if confidence_consistent:
                            logger.info(f"✅ MULTI-FRAME CONFIRMED: {roll_number} - {len(recent_recognitions)} consistent recognitions")
                            return True
                        else:
                            logger.info(f"❌ CONFIDENCE INCONSISTENT: {roll_number}")
                    else:
                        logger.info(f"❌ SPATIAL INCONSISTENT: {roll_number}")
                else:
                    logger.info(f"❌ TEMPORAL INCONSISTENT: {roll_number} - {time_span:.1f}s span")

            logger.info(f"⏳ CONFIRMATION PENDING: {roll_number} - {len(recent_recognitions)}/{self.recognition_confirmation_frames} frames")
            return False

        except Exception as e:
            logger.error(f"Error in multi-frame confirmation: {e}")
            return False

    def _check_spatial_consistency(self, recognitions: List[Dict]) -> bool:
        """Check if face positions are spatially consistent (same person)"""
        try:
            if len(recognitions) < 2:
                return True

            # Calculate center points of all bboxes
            centers = []
            for rec in recognitions:
                bbox = rec['bbox']
                if len(bbox) >= 4:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    centers.append((center_x, center_y))

            if len(centers) < 2:
                return True

            # Check if all centers are within reasonable distance
            for i in range(1, len(centers)):
                distance = ((centers[i][0] - centers[i-1][0])**2 + (centers[i][1] - centers[i-1][1])**2)**0.5
                if distance > self.max_recognition_distance:
                    logger.info(f"❌ SPATIAL DISTANCE TOO LARGE: {distance:.1f} > {self.max_recognition_distance}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking spatial consistency: {e}")
            return False

    def _cleanup_recognition_buffers(self, current_time: float):
        """Clean up old recognition buffers to prevent memory leaks"""
        try:
            cleanup_threshold = 30.0  # Remove buffers older than 30 seconds

            for roll_number in list(self.recognition_buffer.keys()):
                buffer = self.recognition_buffer[roll_number]
                if buffer:
                    # Remove old recognitions
                    self.recognition_buffer[roll_number] = [
                        rec for rec in buffer
                        if current_time - rec['timestamp'] < cleanup_threshold
                    ]

                    # Remove empty buffers
                    if not self.recognition_buffer[roll_number]:
                        del self.recognition_buffer[roll_number]

        except Exception as e:
            logger.error(f"Error cleaning recognition buffers: {e}")

    def _continuous_learning_update(self, face_data: Dict, recognition_result: Dict = None, feedback_type: str = "detection"):
        """Continuous learning update for face detection and recognition improvement"""
        try:
            if not self.continuous_learning_enabled:
                return

            current_time = time.time()

            # Track face detection performance
            face_confidence = face_data.get('confidence', 0.0)
            face_area = face_data.get('bbox', [0, 0, 100, 100])
            if len(face_area) >= 4:
                area = face_area[2] * face_area[3] if face_area[2] < 1000 else (face_area[2] - face_area[0]) * (face_area[3] - face_area[1])
            else:
                area = 10000

            # Create learning instance
            learning_instance = {
                'timestamp': current_time,
                'face_confidence': face_confidence,
                'face_area': area,
                'recognition_confidence': recognition_result.get('confidence', 0.0) if recognition_result else 0.0,
                'recognition_success': recognition_result is not None,
                'feedback_type': feedback_type,
                'threshold_used': self.face_recognition_threshold,
                'session_duration': current_time - self.session_start_time
            }

            # Add to learning buffer
            self.learning_buffer.append(learning_instance)

            # Update performance metrics
            self.total_detections += 1

            if recognition_result:
                self.correct_recognitions += 1
                # Positive reinforcement - slightly lower threshold for similar faces
                self._adjust_threshold_reinforcement(face_confidence, recognition_result['confidence'], reward=1.0)
            else:
                # Negative reinforcement - slightly raise threshold to prevent false positives
                self._adjust_threshold_reinforcement(face_confidence, 0.0, reward=-0.5)

            # Calculate rolling accuracy
            if len(self.learning_buffer) >= 10:
                recent_instances = list(self.learning_buffer)[-10:]
                accuracy = sum(1 for inst in recent_instances if inst['recognition_success']) / len(recent_instances)
                self.detection_accuracy_window.append(accuracy)

            # Adaptive threshold adjustment based on performance
            if len(self.detection_accuracy_window) >= 10:
                avg_accuracy = sum(self.detection_accuracy_window) / len(self.detection_accuracy_window)
                self._adaptive_threshold_adjustment(avg_accuracy)

            # Feed data to external continuous learning system
            if self.external_learning_system and CONTINUOUS_LEARNING_AVAILABLE:
                try:
                    # Create learning instance for external system
                    if recognition_result:
                        feedback_type = FeedbackType.CORRECT_PREDICTION if feedback_type == "successful_recognition" else FeedbackType.SYSTEM_VALIDATION
                        predicted_label = recognition_result['roll_number']
                        actual_label = recognition_result['roll_number']  # Assume correct for successful recognition
                        confidence = recognition_result['confidence']
                    else:
                        feedback_type = FeedbackType.UNCERTAINTY_SAMPLING
                        predicted_label = "unknown"
                        actual_label = None
                        confidence = face_confidence

                    external_learning_instance = LearningInstance(
                        timestamp=current_time,
                        features={
                            'face_confidence': face_confidence,
                            'face_area': area,
                            'threshold_used': self.face_recognition_threshold,
                            'session_duration': current_time - self.session_start_time,
                            'detection_type': feedback_type
                        },
                        predicted_label=predicted_label,
                        actual_label=actual_label,
                        confidence=confidence,
                        feedback_type=feedback_type,
                        model_version="attendance_v1.0",
                        session_id=f"session_{int(self.session_start_time)}",
                        metadata={'component': 'automated_attendance'}
                    )

                    # Add to external learning system
                    self.external_learning_system.add_prediction_feedback(
                        model_name="face_recognition",
                        features=external_learning_instance.features,
                        predicted_label=external_learning_instance.predicted_label,
                        confidence=external_learning_instance.confidence,
                        actual_label=external_learning_instance.actual_label,
                        feedback_type=external_learning_instance.feedback_type
                    )

                except Exception as e:
                    logger.error(f"Error feeding data to external learning system: {e}")

            # Log learning progress periodically
            if self.total_detections % 50 == 0:
                self._log_learning_progress()

            # AUTO-SAVE CHECKPOINT PERIODICALLY
            if self.total_detections - self.last_checkpoint_save >= self.auto_save_interval:
                self._save_learning_checkpoint()
                self.last_checkpoint_save = self.total_detections

        except Exception as e:
            logger.error(f"Error in continuous learning update: {e}")

    def _adjust_threshold_reinforcement(self, face_confidence: float, recognition_confidence: float, reward: float):
        """Reinforcement learning for threshold adjustment"""
        try:
            # Q-learning inspired threshold adjustment
            current_threshold = self.face_recognition_threshold

            # Calculate reward based on detection quality and recognition success
            quality_reward = face_confidence * 0.5  # Higher face confidence = better quality
            recognition_reward = recognition_confidence * reward
            total_reward = quality_reward + recognition_reward

            # Adjust threshold using learning rate and reward
            threshold_adjustment = self.learning_rate * total_reward * self.confidence_adjustment_factor

            # Apply exploration (epsilon-greedy)
            if np.random.random() < self.exploration_rate:
                threshold_adjustment += np.random.normal(0, 0.01)  # Small random exploration

            # Update threshold with bounds
            new_threshold = current_threshold + threshold_adjustment
            new_threshold = max(0.3, min(0.8, new_threshold))  # Keep within reasonable bounds

            # Apply threshold change gradually
            self.face_recognition_threshold = 0.9 * current_threshold + 0.1 * new_threshold

            # Track threshold changes
            self.confidence_threshold_history.append({
                'timestamp': time.time(),
                'old_threshold': current_threshold,
                'new_threshold': self.face_recognition_threshold,
                'reward': total_reward,
                'face_confidence': face_confidence,
                'recognition_confidence': recognition_confidence
            })

        except Exception as e:
            logger.error(f"Error in threshold reinforcement: {e}")

    def _adaptive_threshold_adjustment(self, current_accuracy: float):
        """Adaptive threshold adjustment based on overall performance"""
        try:
            target_accuracy = 0.85  # Target 85% accuracy
            accuracy_diff = current_accuracy - target_accuracy

            # If accuracy is too low, lower threshold to be more lenient
            if accuracy_diff < -0.1:  # More than 10% below target
                adjustment = -0.02
            elif accuracy_diff < -0.05:  # 5-10% below target
                adjustment = -0.01
            # If accuracy is too high, raise threshold to be more strict
            elif accuracy_diff > 0.1:  # More than 10% above target
                adjustment = 0.02
            elif accuracy_diff > 0.05:  # 5-10% above target
                adjustment = 0.01
            else:
                adjustment = 0  # Within acceptable range

            if adjustment != 0:
                old_threshold = self.face_recognition_threshold
                self.face_recognition_threshold = max(0.3, min(0.8, old_threshold + adjustment))
                logger.info(f"🎯 ADAPTIVE THRESHOLD: {old_threshold:.3f} → {self.face_recognition_threshold:.3f} (Accuracy: {current_accuracy:.3f})")

        except Exception as e:
            logger.error(f"Error in adaptive threshold adjustment: {e}")

    def _log_learning_progress(self):
        """Log continuous learning progress"""
        try:
            if self.total_detections == 0:
                return

            accuracy = self.correct_recognitions / self.total_detections
            session_duration = time.time() - self.session_start_time

            # Calculate recent performance
            recent_accuracy = sum(self.detection_accuracy_window) / len(self.detection_accuracy_window) if self.detection_accuracy_window else 0

            logger.info(f"🧠 CONTINUOUS LEARNING PROGRESS:")
            logger.info(f"   📊 Total Detections: {self.total_detections}")
            logger.info(f"   ✅ Overall Accuracy: {accuracy:.3f}")
            logger.info(f"   📈 Recent Accuracy: {recent_accuracy:.3f}")
            logger.info(f"   🎯 Current Threshold: {self.face_recognition_threshold:.3f}")
            logger.info(f"   ⏱️ Session Duration: {session_duration/60:.1f} minutes")
            logger.info(f"   🔄 Learning Buffer Size: {len(self.learning_buffer)}")

        except Exception as e:
            logger.error(f"Error logging learning progress: {e}")

    def _load_learning_checkpoint(self):
        """Load learning checkpoint to continue from last session"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                # Restore learning parameters
                self.face_recognition_threshold = checkpoint_data.get('face_recognition_threshold', self.face_recognition_threshold)
                self.learning_rate = checkpoint_data.get('learning_rate', self.learning_rate)
                self.exploration_rate = checkpoint_data.get('exploration_rate', self.exploration_rate)

                # Restore performance metrics
                self.total_detections = checkpoint_data.get('total_detections', 0)
                self.correct_recognitions = checkpoint_data.get('correct_recognitions', 0)
                self.false_positives = checkpoint_data.get('false_positives', 0)
                self.false_negatives = checkpoint_data.get('false_negatives', 0)

                # Restore learning buffers (convert lists back to deques)
                if 'learning_buffer' in checkpoint_data:
                    self.learning_buffer = deque(checkpoint_data['learning_buffer'], maxlen=1000)
                if 'recognition_history' in checkpoint_data:
                    self.recognition_history = deque(checkpoint_data['recognition_history'], maxlen=500)
                if 'face_detection_history' in checkpoint_data:
                    self.face_detection_history = deque(checkpoint_data['face_detection_history'], maxlen=500)
                if 'confidence_threshold_history' in checkpoint_data:
                    self.confidence_threshold_history = deque(checkpoint_data['confidence_threshold_history'], maxlen=100)
                if 'detection_accuracy_window' in checkpoint_data:
                    self.detection_accuracy_window = deque(checkpoint_data['detection_accuracy_window'], maxlen=50)

                # Restore recognition buffer
                if 'recognition_buffer' in checkpoint_data:
                    self.recognition_buffer = checkpoint_data['recognition_buffer']

                checkpoint_age = time.time() - checkpoint_data.get('save_timestamp', time.time())
                logger.info(f"📂 CHECKPOINT LOADED: {self.total_detections} detections, {self.correct_recognitions} correct recognitions")
                logger.info(f"🎯 RESTORED THRESHOLD: {self.face_recognition_threshold:.3f}")
                logger.info(f"⏰ CHECKPOINT AGE: {checkpoint_age/3600:.1f} hours")

            else:
                logger.info("📂 No checkpoint found - starting fresh learning session")

        except Exception as e:
            logger.error(f"Error loading learning checkpoint: {e}")
            logger.info("📂 Starting fresh learning session due to checkpoint error")

    def _save_learning_checkpoint(self):
        """Save learning checkpoint for persistence across sessions"""
        try:
            checkpoint_data = {
                'save_timestamp': time.time(),
                'face_recognition_threshold': self.face_recognition_threshold,
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'total_detections': self.total_detections,
                'correct_recognitions': self.correct_recognitions,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives,
                'learning_buffer': list(self.learning_buffer),  # Convert deque to list for pickling
                'recognition_history': list(self.recognition_history),
                'face_detection_history': list(self.face_detection_history),
                'confidence_threshold_history': list(self.confidence_threshold_history),
                'detection_accuracy_window': list(self.detection_accuracy_window),
                'recognition_buffer': self.recognition_buffer,
                'session_start_time': self.session_start_time
            }

            # Create backup of existing checkpoint
            if os.path.exists(self.checkpoint_file):
                backup_file = self.checkpoint_file + '.backup'
                shutil.copy2(self.checkpoint_file, backup_file)

            # Save new checkpoint
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            logger.info(f"💾 CHECKPOINT SAVED: {self.total_detections} detections, accuracy: {self.correct_recognitions/max(1, self.total_detections):.3f}")

        except Exception as e:
            logger.error(f"Error saving learning checkpoint: {e}")

    def _reinforcement_learning_update(self, frame: np.ndarray, faces: List[Dict], recognition_results: List[Dict], current_time: float):
        """Continuous reinforcement learning to improve model performance"""
        try:
            if not self.reinforcement_learning_enabled:
                return

            # 1. FACE DETECTION QUALITY ASSESSMENT
            self._assess_detection_quality(faces)

            # 2. RECOGNITION ACCURACY TRACKING
            self._track_recognition_accuracy(recognition_results, faces)

            # 3. ADAPTIVE THRESHOLD OPTIMIZATION
            self._optimize_recognition_thresholds(recognition_results)

            # 4. DETECTION PARAMETER OPTIMIZATION
            self._optimize_detection_parameters(faces)

            # 5. EXPERIENCE BUFFER MANAGEMENT
            self._update_experience_buffer(frame, faces, recognition_results, current_time)

            # 6. PERIODIC MODEL UPDATES
            if current_time - self.learning_metrics['last_model_update'] > 60.0:  # Every minute
                self._perform_model_updates()
                self.learning_metrics['last_model_update'] = current_time

            # 7. UPDATE LEARNING METRICS
            self._update_learning_metrics()

            logger.debug(f"🧠 RL UPDATE: Accuracy: {self._get_current_accuracy():.3f}, Detection Quality: {self._get_detection_quality():.3f}")

        except Exception as e:
            logger.error(f"Error in reinforcement learning update: {e}")

    def _assess_detection_quality(self, faces: List[Dict]):
        """Assess and learn from face detection quality"""
        try:
            for face in faces:
                confidence = face.get('confidence', 0.0)
                bbox = face.get('bbox', [])

                # Calculate quality score based on multiple factors
                quality_score = confidence

                # Bonus for larger faces (better resolution)
                if len(bbox) >= 4:
                    face_area = bbox[2] * bbox[3] if bbox[2] < 1000 else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area_bonus = min(face_area / 50000, 0.2)  # Max 0.2 bonus
                    quality_score += area_bonus

                # Penalty for edge faces (partial visibility)
                if len(bbox) >= 4:
                    x, y = bbox[0], bbox[1]
                    if x < 50 or y < 50:  # Near edges
                        quality_score -= 0.1

                self.detection_quality_scores.append(quality_score)

                # Keep only recent scores
                if len(self.detection_quality_scores) > 100:
                    self.detection_quality_scores.pop(0)

        except Exception as e:
            logger.error(f"Error assessing detection quality: {e}")

    def _track_recognition_accuracy(self, recognition_results: List[Dict], faces: List[Dict]):
        """Track and learn from recognition accuracy"""
        try:
            self.recognition_accuracy_tracker['total_attempts'] += len(faces)

            # Track successful recognitions
            for result in recognition_results:
                confidence = result.get('confidence', 0.0)

                # High confidence recognitions are likely correct
                if confidence > 0.7:
                    self.recognition_accuracy_tracker['correct_recognitions'] += 1
                    self.recognition_success_history.append(True)
                elif confidence > 0.4:
                    # Medium confidence - uncertain
                    pass
                else:
                    # Low confidence - likely false positive
                    self.recognition_accuracy_tracker['false_positives'] += 1
                    self.false_positive_history.append(True)

            # Track faces that weren't recognized (potential false negatives)
            unrecognized_faces = len(faces) - len(recognition_results)
            if unrecognized_faces > 0:
                self.recognition_accuracy_tracker['false_negatives'] += unrecognized_faces

            # Keep history manageable
            if len(self.recognition_success_history) > 200:
                self.recognition_success_history.pop(0)
            if len(self.false_positive_history) > 200:
                self.false_positive_history.pop(0)

        except Exception as e:
            logger.error(f"Error tracking recognition accuracy: {e}")

    def _optimize_recognition_thresholds(self, recognition_results: List[Dict]):
        """Dynamically optimize recognition thresholds based on performance"""
        try:
            if not recognition_results:
                return

            # Calculate current performance metrics
            recent_accuracy = self._get_recent_accuracy()
            recent_false_positive_rate = self._get_recent_false_positive_rate()

            # Adaptive threshold adjustment
            current_threshold = self.face_recognition_threshold

            if recent_false_positive_rate > 0.2:  # Too many false positives
                # Increase threshold to be more strict
                new_threshold = min(current_threshold + 0.01, 0.8)
                logger.info(f"🧠 RL: Increasing threshold {current_threshold:.3f} → {new_threshold:.3f} (reducing false positives)")
            elif recent_accuracy > 0.8 and recent_false_positive_rate < 0.05:  # Good performance
                # Slightly decrease threshold to catch more faces
                new_threshold = max(current_threshold - 0.005, 0.3)
                logger.info(f"🧠 RL: Decreasing threshold {current_threshold:.3f} → {new_threshold:.3f} (improving sensitivity)")
            else:
                new_threshold = current_threshold

            self.face_recognition_threshold = new_threshold
            self.threshold_history.append(new_threshold)

            # Keep history manageable
            if len(self.threshold_history) > 100:
                self.threshold_history.pop(0)

        except Exception as e:
            logger.error(f"Error optimizing recognition thresholds: {e}")

    def _optimize_detection_parameters(self, faces: List[Dict]):
        """Optimize face detection parameters based on performance"""
        try:
            if not faces:
                return

            # Analyze detection quality
            avg_confidence = sum(face.get('confidence', 0.0) for face in faces) / len(faces)

            # Adjust detection confidence threshold
            if avg_confidence < 0.6:  # Low average confidence
                # Lower threshold to detect more faces
                self.optimal_detection_params['confidence_threshold'] = max(
                    self.optimal_detection_params['confidence_threshold'] - 0.01, 0.3
                )
            elif avg_confidence > 0.9:  # Very high confidence
                # Raise threshold to reduce false detections
                self.optimal_detection_params['confidence_threshold'] = min(
                    self.optimal_detection_params['confidence_threshold'] + 0.01, 0.8
                )

            logger.debug(f"🧠 RL: Detection params updated - confidence_threshold: {self.optimal_detection_params['confidence_threshold']:.3f}")

        except Exception as e:
            logger.error(f"Error optimizing detection parameters: {e}")

    def _lock_face_immediately(self, recognition_result: Dict, current_time: float):
        """Lock a face immediately when recognized for persistent labeling"""
        try:
            roll_number = recognition_result['roll_number']
            person_id = f"locked_{roll_number}"

            # Create locked face entry
            self.locked_faces[person_id] = {
                'roll_number': roll_number,
                'name': recognition_result['name'],
                'face_bbox': recognition_result['face_bbox'],
                'confidence': recognition_result['confidence'],
                'lock_time': current_time,
                'last_seen': current_time,
                'is_locked': True,
                'lock_duration': 0.0
            }

            logger.info(f"🔒 FACE LOCKED: {recognition_result['name']} ({roll_number}) - Immediate lock activated!")

        except Exception as e:
            logger.error(f"Error locking face: {e}")

    def _update_locked_faces(self, current_time: float, detected_faces: List[Dict]):
        """Update locked faces and maintain persistence"""
        try:
            faces_to_remove = []

            for person_id, locked_face in self.locked_faces.items():
                # Check if this locked face is still being detected
                face_still_detected = False

                for detected_face in detected_faces:
                    if detected_face.get('roll_number') == locked_face['roll_number']:
                        # Update locked face with new detection
                        locked_face['face_bbox'] = detected_face['face_bbox']
                        locked_face['confidence'] = detected_face['confidence']
                        locked_face['last_seen'] = current_time
                        face_still_detected = True
                        break

                # Calculate lock duration
                locked_face['lock_duration'] = current_time - locked_face['lock_time']

                # Remove locked face if not seen for too long
                time_since_last_seen = current_time - locked_face['last_seen']
                if time_since_last_seen > self.face_lock_timeout:
                    faces_to_remove.append(person_id)
                    logger.info(f"🔓 FACE UNLOCKED: {locked_face['name']} - Lock timeout ({time_since_last_seen:.1f}s)")
                elif face_still_detected:
                    logger.debug(f"🔒 FACE LOCKED: {locked_face['name']} - Duration: {locked_face['lock_duration']:.1f}s")

            # Remove expired locked faces
            for person_id in faces_to_remove:
                del self.locked_faces[person_id]

        except Exception as e:
            logger.error(f"Error updating locked faces: {e}")

    def get_latest_result(self):
        """Get the latest attendance result for display"""
        return self.latest_result

    def _recognize_faces(self, frame: np.ndarray, faces: List[Dict], current_time: float = None) -> List[Dict]:
        """Recognize faces using enhanced DeepFace with industrial-grade accuracy"""
        if current_time is None:
            current_time = time.time()
        recognition_results = []

        if not DEEPFACE_AVAILABLE:
            logger.warning("DeepFace not available. Skipping recognition.")
            return recognition_results

        if not self.known_face_encodings:
            logger.warning(f"No known face encodings available. Students loaded: {len(self.students_db)}")
            return recognition_results

        logger.info(f"🔍 INDUSTRIAL-GRADE RECOGNITION: Processing {len(faces)} faces. Known encodings: {len(self.known_face_encodings)}")
        logger.info(f"🔍 FACE DATA STRUCTURE: {faces}")

        try:
            for i, face in enumerate(faces):
                logger.info(f"🔍 PROCESSING FACE {i}: {face}")
                bbox = face.get('bbox', [])
                logger.info(f"🔍 BBOX: {bbox}, length: {len(bbox)}")
                if len(bbox) != 4:
                    logger.warning(f"❌ INVALID BBOX: Expected 4 values, got {len(bbox)}")
                    continue

                # Convert bbox from [x, y, width, height] to [x1, y1, x2, y2] format
                # Face detection returns [x, y, width, height] format, we need [x1, y1, x2, y2]
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                logger.info(f"🔍 BBOX CONVERSION: [x={x}, y={y}, w={w}, h={h}] -> [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")

                # STRICT VALIDATION: Bbox coordinates
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"❌ INVALID BBOX DIMENSIONS: x2({x2}) <= x1({x1}) or y2({y2}) <= y1({y1})")
                    continue

                # ULTRA-AGGRESSIVE VALIDATION: Detect even the tiniest heads
                face_confidence = face.get('confidence', 0.0)
                original_confidence = face.get('original_confidence', face_confidence)
                is_micro_face = face.get('is_micro_face', False)
                is_tiny_face = face.get('is_tiny_face', False)
                is_small_face = face.get('is_small_face', False)
                face_quality = face.get('face_quality', 'unknown')
                size_category = face.get('size_category', 'unknown')
                face_size = face.get('face_size', 0)

                # EXTREMELY low threshold for maximum detection - catch even 6x6 pixel faces
                if is_micro_face:
                    min_confidence = 0.01  # Ultra-low for micro faces
                elif is_tiny_face:
                    min_confidence = 0.02  # Very low for tiny faces
                elif is_small_face:
                    min_confidence = 0.03  # Low for small faces
                else:
                    min_confidence = 0.05  # Standard low threshold

                if face_confidence < min_confidence:
                    # CONTINUOUS LEARNING: Track ultra-low confidence detections
                    self._continuous_learning_update(face, None, f"ultra_low_confidence_{size_category}")
                    logger.debug(f"❌ ULTRA LOW CONFIDENCE: {face_confidence:.3f} < {min_confidence:.3f} ({size_category})")
                    continue

                # ULTRA-PERMISSIVE AREA VALIDATION: Accept even 6x6 pixel faces
                face_area = w * h
                if is_micro_face:
                    min_area = 36   # 6x6 pixels minimum
                elif is_tiny_face:
                    min_area = 64   # 8x8 pixels minimum
                elif is_small_face:
                    min_area = 100  # 10x10 pixels minimum
                else:
                    min_area = 200  # Standard minimum

                if face_area < min_area:
                    # CONTINUOUS LEARNING: Track ultra-tiny face detections
                    self._continuous_learning_update(face, None, f"ultra_tiny_face_{size_category}")
                    logger.debug(f"❌ FACE TOO TINY: Area {face_area} < {min_area} ({size_category})")
                    continue

                # CONTINUOUS LEARNING: Track valid face detection with detailed info
                self._continuous_learning_update(face, None, f"valid_face_{size_category}_{face_quality}")

                # Log detection details for monitoring tiny faces
                logger.info(f"✅ FACE DETECTED: Size={face_size}px, Conf={face_confidence:.3f}, Area={face_area}, Category={size_category}, Quality={face_quality}")

                # Extract face region with minimal padding (reduce noise)
                padding = 10  # Reduced padding to minimize background noise
                y1_padded = max(0, y1 - padding)
                y2_padded = min(frame.shape[0], y2 + padding)
                x1_padded = max(0, x1 - padding)
                x2_padded = min(frame.shape[1], x2 + padding)

                face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                logger.info(f"🔍 FACE REGION EXTRACTED: Shape {face_region.shape}, Area: {face_area}, Confidence: {face_confidence:.3f}")
                if face_region.size == 0 or face_region.shape[0] < 50 or face_region.shape[1] < 50:
                    continue

                try:
                    logger.info(f"🔍 PROCESSING: Face region {face_region.shape} at bbox {bbox}")

                    # INDUSTRIAL-GRADE PREPROCESSING
                    preprocessed_face = self._preprocess_face_for_recognition(face_region)
                    logger.info(f"✅ PREPROCESSING: Success, preprocessed shape {preprocessed_face.shape}")

                    # SINGLE-MODEL RECOGNITION for optimal performance and stability
                    face_embeddings = []
                    model_name = 'Facenet'  # Use only Facenet
                    logger.info(f"🔍 USING MODEL: {model_name} only for recognition")

                    try:
                            embedding_objs = DeepFace.represent(
                                img_path=preprocessed_face,
                                model_name=model_name,
                                enforce_detection=False,  # More lenient for cropped faces
                                detector_backend='opencv'
                            )

                            if embedding_objs and len(embedding_objs) > 0:
                                face_embeddings.append({
                                    'model': model_name,
                                    'embedding': np.array(embedding_objs[0]['embedding'])
                                })
                                logger.info(f"✅ {model_name} embedding generated: {len(embedding_objs[0]['embedding'])} dims")
                    except Exception as model_error:
                        logger.error(f"❌ {model_name} failed: {model_error}")

                    if not face_embeddings:
                        logger.error("❌ NO EMBEDDINGS: No embeddings generated from any model")
                        continue

                    logger.info(f"🔍 EMBEDDINGS SUCCESS: Generated {len(face_embeddings)} embeddings")

                    # ENSEMBLE RECOGNITION: Compare with known faces using multiple models
                    best_match_roll = None
                    best_confidence = 0.0
                    all_similarities = []
                    best_embedding = None

                    # For each known student, calculate similarity across all models
                    for roll_number, known_embedding in zip(self.known_roll_numbers, self.known_face_encodings):
                        model_similarities = []

                        # Compare with each model's embedding
                        for face_emb_data in face_embeddings:
                            similarity = self._cosine_similarity(face_emb_data['embedding'], known_embedding)
                            model_similarities.append(similarity)

                        # Use ensemble approach: average of all models or best model
                        if model_similarities:
                            # Take the maximum similarity across models for robustness
                            max_similarity = max(model_similarities)
                            avg_similarity = sum(model_similarities) / len(model_similarities)

                            # Use weighted combination: 70% max, 30% average
                            final_similarity = 0.7 * max_similarity + 0.3 * avg_similarity

                            all_similarities.append(f"{roll_number}:{final_similarity:.3f}")

                            # Use stricter base threshold (no adaptive lowering for now)
                            threshold = self.face_recognition_threshold

                            if final_similarity > best_confidence and final_similarity > threshold:
                                best_confidence = final_similarity
                                best_match_roll = roll_number
                                # Use the embedding from the best performing model
                                best_model_idx = model_similarities.index(max_similarity)
                                best_embedding = face_embeddings[best_model_idx]['embedding']

                    # DETAILED LOGGING FOR DEBUGGING
                    max_sim = max([float(s.split(':')[1]) for s in all_similarities]) if all_similarities else 0.0
                    logger.info(f"🔍 ENSEMBLE similarities: {', '.join(all_similarities)} | Threshold: {self.face_recognition_threshold:.3f} | Max: {max_sim:.3f}")

                    if best_match_roll:
                        # MULTI-FRAME CONFIRMATION: Prevent false positives
                        confirmed_recognition = self._confirm_recognition_multi_frame(
                            best_match_roll, best_confidence, bbox, current_time
                        )

                        if confirmed_recognition:
                            student = self.students_db[best_match_roll]

                            recognition_result = {
                                'face_bbox': bbox,
                                'roll_number': best_match_roll,
                                'name': student.name,
                                'confidence': best_confidence,
                                'face_encoding': best_embedding,
                                'models_used': len(face_embeddings)
                            }

                            recognition_results.append(recognition_result)

                            # CONTINUOUS LEARNING: Positive feedback for successful recognition
                            self._continuous_learning_update(face, recognition_result, "successful_recognition")

                            # IMMEDIATE FACE LOCKING - Lock the face as soon as confirmed
                            if self.immediate_lock_enabled:
                                self._lock_face_immediately(recognition_result, current_time)

                            logger.info(f"✅ CONFIRMED RECOGNITION: {student.name} ({best_match_roll}) - Confidence: {best_confidence:.3f} (Models: {len(face_embeddings)}) - LOCKED!")
                        else:
                            # CONTINUOUS LEARNING: Pending confirmation feedback
                            self._continuous_learning_update(face, None, "pending_confirmation")
                            logger.info(f"⏳ PENDING CONFIRMATION: {best_match_roll} - Confidence: {best_confidence:.3f} (Need more frames)")
                    else:
                        # CONTINUOUS LEARNING: No match feedback
                        self._continuous_learning_update(face, None, "no_match")

                        # TRACK UNKNOWN PERSON if enabled
                        if self.track_unknown_people:
                            unknown_person_result = self._handle_unknown_person(face, bbox, best_embedding, max_sim, current_time)
                            if unknown_person_result:
                                recognition_results.append(unknown_person_result)

                        logger.info(f"❌ No match found. Best similarity: {max_sim:.3f} < Threshold: {self.face_recognition_threshold:.3f} | All: {', '.join(all_similarities)}")

                except Exception as deepface_error:
                    logger.warning(f"DeepFace recognition failed for face: {deepface_error}")
                    continue

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
        finally:
            # Clean up old recognition buffers
            self._cleanup_recognition_buffers(current_time)

        return recognition_results

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _update_tracking(self, faces: List[Dict], recognition_results: List[Dict], current_time: float):
        """Update tracking for all detected persons"""
        try:
            # Mark all tracked persons as potentially missing
            for person in self.tracked_persons.values():
                person.is_present = False

            # Process recognized faces first
            for result in recognition_results:
                self._process_recognized_face(result, current_time)

            # Process unrecognized faces
            recognized_bboxes = [r['face_bbox'] for r in recognition_results]
            for face in faces:
                bbox = face.get('bbox', [])
                if bbox not in recognized_bboxes:
                    self._process_unrecognized_face(face, current_time)

            # Handle missing persons
            missing_persons = [p for p in self.tracked_persons.values() if not p.is_present]
            for person in missing_persons:
                self._handle_missing_person(person, current_time)

        except Exception as e:
            logger.error(f"Error in tracking update: {e}")

    def _process_recognized_face(self, result: Dict, current_time: float):
        """Process a recognized face for tracking"""
        roll_number = result['roll_number']
        bbox = result['face_bbox']

        # Find existing tracked person or create new one
        existing_person = None
        for person in self.tracked_persons.values():
            if person.roll_number == roll_number:
                existing_person = person
                break

        if existing_person:
            # Update existing tracking
            existing_person.face_bbox = bbox
            existing_person.last_seen = current_time
            existing_person.tracking_confidence = result['confidence']
            existing_person.is_present = True
            existing_person.alert_triggered = False
            existing_person.alert_start_time = None

            # Remove any active alerts for this person
            if existing_person.person_id in self.active_alerts:
                del self.active_alerts[existing_person.person_id]
        else:
            # Create new tracked person
            person_id = f"person_{self.next_person_id}"
            self.next_person_id += 1

            tracked_person = TrackedPerson(
                person_id=person_id,
                roll_number=roll_number,
                name=result['name'],
                face_bbox=bbox,
                face_encoding=result['face_encoding'],
                last_seen=current_time,
                tracking_confidence=result['confidence'],
                entry_time=datetime.now(),
                is_present=True
            )

            self.tracked_persons[person_id] = tracked_person

            # Create attendance record ONLY for known students (not unknown people)
            if (roll_number not in self.attendance_records and
                not roll_number.startswith('unknown_') and
                not roll_number.startswith('UNKNOWN_') and
                roll_number in self.students_db):
                self.attendance_records[roll_number] = AttendanceRecord(
                    roll_number=roll_number,
                    name=result['name'],
                    entry_time=datetime.now()
                )

                # Log attendance
                self._log_attendance_entry(roll_number, result['name'])
                logger.info(f"📝 ATTENDANCE: {result['name']} ({roll_number}) marked present")
            elif (roll_number.startswith('unknown_') or roll_number.startswith('UNKNOWN_')):
                logger.debug(f"👤 Unknown person detected but not added to attendance: {roll_number}")

    def _process_unrecognized_face(self, face: Dict, current_time: float):
        """Process an unrecognized face for basic tracking"""
        bbox = face.get('bbox', [])

        # Try to match with existing unrecognized persons by proximity
        matched_person = None
        min_distance = float('inf')

        for person in self.tracked_persons.values():
            if person.roll_number.startswith('unknown_'):
                distance = self._calculate_bbox_distance(bbox, person.face_bbox)
                if distance < self.max_tracking_distance and distance < min_distance:
                    min_distance = distance
                    matched_person = person

        if matched_person:
            # Update existing unrecognized person
            matched_person.face_bbox = bbox
            matched_person.last_seen = current_time
            matched_person.is_present = True
        else:
            # Create new unrecognized person
            person_id = f"person_{self.next_person_id}"
            self.next_person_id += 1

            tracked_person = TrackedPerson(
                person_id=person_id,
                roll_number=f"unknown_{person_id}",
                name="Unknown Person",
                face_bbox=bbox,
                face_encoding=np.array([]),
                last_seen=current_time,
                tracking_confidence=face.get('confidence', 0.5),
                entry_time=datetime.now(),
                is_present=True
            )

            self.tracked_persons[person_id] = tracked_person

    def _handle_missing_person(self, person: TrackedPerson, current_time: float):
        """Handle a person who has disappeared from view"""
        time_missing = current_time - person.last_seen

        # STRICT ALERT POLICY: Only alert for KNOWN STUDENTS in database
        # Skip alerts for unknown persons, unrecognized faces, or invalid roll numbers
        if (person.roll_number.startswith('unknown_') or
            person.roll_number.startswith('UNKNOWN_') or
            person.roll_number not in self.students_db):
            logger.debug(f"Skipping alert for unknown/unrecognized person: {person.person_id} ({person.roll_number})")
            return

        logger.debug(f"Checking missing student: {person.name} ({person.roll_number}) - Missing for {time_missing:.1f}s")

        if not person.alert_triggered and time_missing > 5.0:  # 5 seconds grace period
            person.alert_triggered = True
            person.alert_start_time = current_time

            # Create disappearance alert ONLY for recognized students
            alert = {
                'person_id': person.person_id,
                'roll_number': person.roll_number,
                'name': person.name,
                'alert_type': 'disappearance',
                'message': f"Student {person.name} has disappeared from view",
                'start_time': current_time,
                'duration': 0.0
            }

            self.active_alerts[person.person_id] = alert
            self._log_alert(person.roll_number, 'disappearance', alert['message'])
            logger.warning(f"🚨 STUDENT ALERT: {person.name} ({person.roll_number}) disappeared from view")

        elif person.alert_triggered and person.alert_start_time:
            alert_duration = current_time - person.alert_start_time

            if person.person_id in self.active_alerts:
                self.active_alerts[person.person_id]['duration'] = alert_duration

            # Remove alert after 30 seconds
            if alert_duration > self.disappearance_alert_duration:
                if person.person_id in self.active_alerts:
                    del self.active_alerts[person.person_id]

                # Mark as exited if recognized student
                if not person.roll_number.startswith('unknown_'):
                    self._mark_student_exit(person.roll_number, current_time)

                # Remove from tracking
                if person.person_id in self.tracked_persons:
                    del self.tracked_persons[person.person_id]

                logger.info(f"📤 EXIT: {person.name} ({person.roll_number}) marked as exited")

    def _check_disappearances(self, current_time: float):
        """Check for student disappearances and manage alerts"""
        for person in list(self.tracked_persons.values()):
            if not person.is_present:
                self._handle_missing_person(person, current_time)

    def _update_attendance_records(self, current_time: float):
        """Update attendance records with engagement data"""
        for roll_number, record in self.attendance_records.items():
            if record.exit_time is None:  # Still present
                # Calculate duration
                duration = (datetime.now() - record.entry_time).total_seconds() / 60.0  # minutes
                record.total_duration = duration

    def _mark_student_exit(self, roll_number: str, current_time: float):
        """Mark a student as exited"""
        if roll_number in self.attendance_records:
            record = self.attendance_records[roll_number]
            record.exit_time = datetime.now()
            record.total_duration = (record.exit_time - record.entry_time).total_seconds() / 60.0

            # Save to database
            self._save_attendance_record(record)

    def _calculate_bbox_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate distance between two bounding boxes"""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return float('inf')

        # Calculate center points
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

        # Euclidean distance
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        return distance

    def _get_tracked_persons_summary(self) -> List[Dict]:
        """Get summary of all tracked persons"""
        summary = []
        for person in self.tracked_persons.values():
            summary.append({
                'person_id': person.person_id,
                'roll_number': person.roll_number,
                'name': person.name,
                'is_present': person.is_present,
                'confidence': person.tracking_confidence,
                'entry_time': person.entry_time.isoformat(),
                'duration_minutes': (datetime.now() - person.entry_time).total_seconds() / 60.0
            })
        return summary

    def _get_frame_annotations(self) -> List[Dict]:
        """Get annotations for frame display"""
        annotations = []

        # FIRST: Add annotations for ALL DETECTED FACES (for continuous face box detection)
        if hasattr(self, 'current_faces') and self.current_faces:
            for i, face in enumerate(self.current_faces):
                bbox = face.get('bbox', [])
                if len(bbox) == 4:
                    # Convert bbox format if needed
                    if len(bbox) == 4 and bbox[2] < bbox[0]:  # w,h format
                        x, y, w, h = bbox
                        bbox = [x, y, x + w, y + h]

                    face_confidence = face.get('confidence', 0.0)

                    # Default to unknown person for unrecognized faces
                    annotations.append({
                        'type': 'rectangle',
                        'bbox': bbox,
                        'color': (0, 255, 255),  # Yellow for unrecognized faces
                        'thickness': 2,
                        'label': f"Detecting... ({face_confidence:.2f})",
                        'confidence': face_confidence
                    })

        # SECOND: Add annotations for tracked persons (will override face detections)
        for person in self.tracked_persons.values():
            if person.is_present:
                # Use green for recognized students, yellow for unknown
                color = (0, 255, 0) if not person.roll_number.startswith('unknown_') else (0, 255, 255)

                # Use body bbox if available, otherwise face bbox
                bbox = person.face_bbox
                if hasattr(person, 'body_bbox') and person.body_bbox:
                    bbox = person.body_bbox

                annotations.append({
                    'type': 'rectangle',
                    'bbox': bbox,
                    'color': color,
                    'thickness': 3,  # Thicker for better visibility
                    'label': f"{person.name} ({person.roll_number})",
                    'confidence': person.confidence
                })

        # Add annotations for body tracking (if active)
        for person_id, body_info in self.tracked_bodies.items():
            # Convert body bbox to face bbox format for display
            x, y, w, h = body_info['bbox']
            bbox = [int(x), int(y), int(x + w), int(y + h)]

            annotations.append({
                'type': 'rectangle',
                'bbox': bbox,
                'color': (0, 255, 0),  # Green for body tracking
                'thickness': 3,
                'label': f"{body_info['name']} ({body_info['roll_number']}) [BODY]",
                'confidence': body_info['confidence']
            })

        # Add annotations for LOCKED FACES (highest priority - always visible)
        for person_id, locked_face in self.locked_faces.items():
            bbox = locked_face['face_bbox']

            annotations.append({
                'type': 'rectangle',
                'bbox': bbox,
                'color': (0, 0, 255),  # RED for locked faces - highest visibility
                'thickness': 4,  # Thicker border for locked faces
                'label': f"🔒 {locked_face['name']} ({locked_face['roll_number']}) [LOCKED]",
                'confidence': locked_face['confidence'],
                'is_locked': True,
                'lock_duration': locked_face.get('lock_duration', 0.0)
            })

        return annotations

    def _log_attendance_entry(self, roll_number: str, name: str):
        """Log attendance entry to database ONLY for known students"""
        try:
            # STRICT POLICY: Only log attendance for known students in database
            if (roll_number.startswith('unknown_') or
                roll_number.startswith('UNKNOWN_') or
                roll_number not in self.students_db):
                logger.debug(f"Skipping attendance log for unknown person: {roll_number}")
                return

            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    INSERT INTO attendance (roll_number, name, date, entry_time)
                    VALUES (?, ?, ?, ?)
                ''', (roll_number, name, datetime.now().date(), datetime.now()))
                self.db_connection.commit()
                logger.debug(f"Logged attendance entry for {name} ({roll_number})")
        except Exception as e:
            logger.error(f"Failed to log attendance entry: {e}")

    def _save_attendance_record(self, record: AttendanceRecord):
        """Save complete attendance record to database"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    UPDATE attendance
                    SET exit_time = ?, total_duration = ?, engagement_score = ?,
                        participation_score = ?, attention_score = ?
                    WHERE roll_number = ? AND date = ? AND exit_time IS NULL
                ''', (
                    record.exit_time, record.total_duration, record.engagement_score,
                    record.participation_score, record.attention_score,
                    record.roll_number, record.entry_time.date()
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to save attendance record: {e}")

    def _log_alert(self, roll_number: str, alert_type: str, message: str):
        """Log alert to database ONLY for known students"""
        try:
            # STRICT POLICY: Only log alerts for known students in database
            if (roll_number.startswith('unknown_') or
                roll_number.startswith('UNKNOWN_') or
                roll_number not in self.students_db):
                logger.debug(f"Skipping alert log for unknown person: {roll_number}")
                return

            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    INSERT INTO alerts (roll_number, alert_type, message)
                    VALUES (?, ?, ?)
                ''', (roll_number, alert_type, message))
                self.db_connection.commit()
                logger.debug(f"Logged alert for {roll_number}: {alert_type}")
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    def get_attendance_summary(self) -> Dict[str, Any]:
        """Get current attendance summary - SEPARATE known students from unknown people"""
        # Count ONLY known students (not unknown people)
        known_present = len([p for p in self.tracked_persons.values()
                           if p.is_present and
                           not p.roll_number.startswith('unknown_') and
                           not p.roll_number.startswith('UNKNOWN_') and
                           p.roll_number in self.students_db])

        # Count unknown people separately
        unknown_count = len([p for p in self.tracked_persons.values()
                           if p.is_present and
                           (p.roll_number.startswith('unknown_') or p.roll_number.startswith('UNKNOWN_'))])

        # Add unknown people from the dedicated tracking
        unknown_count += len([p for p in self.unknown_persons.values()
                            if p.get('tracking_status') == 'active'])

        total_students = len(self.students_db)
        total_recognized = len(self.attendance_records)

        return {
            'total_students_in_database': total_students,
            'known_students_present': known_present,
            'unknown_people_detected': unknown_count,
            'known_students_absent': total_students - known_present,
            'attendance_rate_known_students': (known_present / total_students * 100) if total_students > 0 else 0,
            'total_recognized': total_recognized,
            'active_alerts_for_known_students': len(self.active_alerts),
            'total_people_in_frame': known_present + unknown_count,
            'attendance_records': list(self.attendance_records.values())
        }

    def add_student_to_dataset(self, roll_number: str, name: str, application_number: str,
                              photo_path: str, department: str = "", year: str = "", section: str = ""):
        """Add a new student to the dataset using DeepFace"""
        try:
            if not DEEPFACE_AVAILABLE:
                logger.error("DeepFace not available. Cannot add student.")
                return False

            # Generate face encoding using DeepFace
            if os.path.exists(photo_path):
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=photo_path,
                        model_name='Facenet',
                        enforce_detection=True,
                        detector_backend='opencv'
                    )

                    if embedding_objs and len(embedding_objs) > 0:
                        face_encoding = np.array(embedding_objs[0]['embedding'])

                        # Create student object
                        student = Student(
                            roll_number=roll_number,
                            name=name,
                            application_number=application_number,
                            photo_path=photo_path,
                            face_encoding=face_encoding,
                            department=department,
                            year=year,
                            section=section
                        )

                        # Add to database
                        self.students_db[roll_number] = student
                        self.known_face_encodings.append(face_encoding)
                        self.known_roll_numbers.append(roll_number)

                        # Save to database
                        if self.db_connection:
                            cursor = self.db_connection.cursor()
                            cursor.execute('''
                                INSERT OR REPLACE INTO students
                                (roll_number, name, application_number, department, year, section, photo_path, face_encoding)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (roll_number, name, application_number, department, year, section,
                                  photo_path, pickle.dumps(face_encoding)))
                            self.db_connection.commit()

                        logger.info(f"Added student using DeepFace: {name} ({roll_number})")
                        return True
                    else:
                        logger.error(f"No face found in photo for {name}")
                        return False

                except Exception as deepface_error:
                    logger.error(f"DeepFace failed for {name}: {deepface_error}")
                    return False
            else:
                logger.error(f"Photo not found: {photo_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to add student: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Save any pending attendance records
            for record in self.attendance_records.values():
                if record.exit_time is None:
                    record.exit_time = datetime.now()
                    record.total_duration = (record.exit_time - record.entry_time).total_seconds() / 60.0
                    self._save_attendance_record(record)

            # Save final checkpoint before shutdown
            self._save_learning_checkpoint()
            logger.info("💾 Final checkpoint saved during cleanup")

            # Close database connection
            if self.db_connection:
                self.db_connection.close()

            logger.info("Automated attendance system cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _handle_unknown_person(self, face, bbox, face_embedding, similarity_score, current_time):
        """Handle detection of unknown person with comprehensive tracking and parameter calculation"""
        try:
            # Generate unique ID for unknown person
            unknown_id = f"UNKNOWN_{self.next_unknown_id:03d}"
            self.next_unknown_id += 1

            # Calculate comprehensive parameters for unknown person
            face_confidence = face.get('confidence', 0.0)
            face_quality = face.get('face_quality', 'unknown')
            is_small_face = face.get('is_small_face', False)
            face_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0

            # Calculate engagement-related parameters
            engagement_score = self._calculate_unknown_person_engagement(face, bbox, face_confidence)
            attention_score = self._calculate_unknown_person_attention(face, bbox)
            participation_score = self._calculate_unknown_person_participation(face, bbox, face_confidence)

            # Create comprehensive unknown person record
            unknown_person_data = {
                'unknown_id': unknown_id,
                'face_bbox': bbox,
                'confidence': face_confidence,
                'similarity_to_known': similarity_score,
                'face_quality': face_quality,
                'is_small_face': is_small_face,
                'face_area': face_area,
                'face_embedding': face_embedding,
                'detection_time': current_time,
                'engagement_score': engagement_score,
                'attention_score': attention_score,
                'participation_score': participation_score,
                'overall_score': (engagement_score + attention_score + participation_score) / 3.0,
                'tracking_status': 'active',
                'first_seen': current_time,
                'last_seen': current_time,
                'detection_count': 1
            }

            # Store in unknown persons tracking
            self.unknown_persons[unknown_id] = unknown_person_data
            self.unknown_person_parameters[unknown_id] = unknown_person_data

            # Create result for display
            recognition_result = {
                'face_bbox': bbox,
                'roll_number': unknown_id,
                'name': f"Unknown Person {self.next_unknown_id-1}",
                'confidence': face_confidence,
                'face_encoding': face_embedding,
                'is_unknown': True,
                'engagement_score': engagement_score,
                'attention_score': attention_score,
                'participation_score': participation_score,
                'overall_score': unknown_person_data['overall_score'],
                'face_quality': face_quality
            }

            logger.info(f"👤 UNKNOWN PERSON DETECTED: {unknown_id} - Conf: {face_confidence:.3f}, Quality: {face_quality}, Engagement: {engagement_score:.2f}")

            return recognition_result

        except Exception as e:
            logger.error(f"Error handling unknown person: {e}")
            return None

    def _calculate_unknown_person_engagement(self, face, bbox, confidence):
        """Calculate engagement score for unknown person based on face characteristics"""
        try:
            base_score = min(confidence * 100, 85)  # Base score from detection confidence

            # Adjust based on face quality
            face_quality = face.get('face_quality', 'unknown')
            quality_multiplier = {
                'excellent': 1.0,
                'good': 0.9,
                'fair': 0.8,
                'poor': 0.7,
                'very_poor': 0.6,
                'unknown': 0.75
            }.get(face_quality, 0.75)

            # Adjust based on face size (larger faces generally indicate more engagement)
            face_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            size_multiplier = min(1.0, face_area / 5000.0) * 0.2 + 0.8  # 0.8 to 1.0 range

            engagement_score = base_score * quality_multiplier * size_multiplier
            return max(0, min(100, engagement_score))

        except Exception as e:
            logger.error(f"Error calculating unknown person engagement: {e}")
            return 50.0  # Default moderate engagement

    def _calculate_unknown_person_attention(self, face, bbox):
        """Calculate attention score for unknown person"""
        try:
            # Base attention score from face detection
            base_attention = 70.0

            # Adjust based on face position (center of frame = higher attention)
            if len(bbox) >= 4:
                face_center_x = bbox[0] + bbox[2] / 2
                face_center_y = bbox[1] + bbox[3] / 2

                # Assume frame dimensions (will be updated with actual frame size)
                frame_center_x = 320  # Default, should be updated with actual frame width/2
                frame_center_y = 240  # Default, should be updated with actual frame height/2

                # Distance from center (normalized)
                distance_from_center = ((face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2)**0.5
                max_distance = (frame_center_x**2 + frame_center_y**2)**0.5
                center_score = max(0, 1.0 - (distance_from_center / max_distance)) * 30

                attention_score = base_attention + center_score
            else:
                attention_score = base_attention

            return max(0, min(100, attention_score))

        except Exception as e:
            logger.error(f"Error calculating unknown person attention: {e}")
            return 70.0  # Default attention score

    def _calculate_unknown_person_participation(self, face, bbox, confidence):
        """Calculate participation score for unknown person"""
        try:
            # Base participation from detection confidence
            base_participation = confidence * 80

            # Adjust based on face size (larger faces may indicate active participation)
            face_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            size_bonus = min(20, face_area / 1000.0)  # Up to 20 points bonus for larger faces

            participation_score = base_participation + size_bonus
            return max(0, min(100, participation_score))

        except Exception as e:
            logger.error(f"Error calculating unknown person participation: {e}")
            return 60.0  # Default participation score

    def _intelligent_detection_coordinator(self, data: Dict[str, Any], frame, current_time: float) -> Dict[str, Any]:
        """INTELLIGENT COORDINATOR: Decides between face, body, or fusion detection based on conditions"""
        try:
            # Get available detection data
            face_results = data.get('face_detection', {})
            faces = face_results.get('faces', [])

            # Get body detection data (simulate if not available)
            body_results = data.get('body_detection', {})
            bodies = body_results.get('bodies', [])

            # If no body detection available, generate it from face detection
            if not bodies and self.body_detection_enabled:
                bodies = self._generate_body_detection_from_faces(faces, frame)

            # INTELLIGENT ANALYSIS: Determine best detection strategy
            detection_strategy = self._analyze_detection_strategy(faces, bodies, frame)

            recognition_results = []
            detection_stats = {
                'detection_method': detection_strategy,
                'total_detections': 0,
                'face_detections': 0,
                'body_detections': 0,
                'fusion_detections': 0
            }

            if detection_strategy == "face_only":
                # Close people: Use face detection only
                if faces and self.frame_count % self.face_recognition_interval == 0:
                    recognition_results = self._recognize_faces(frame, faces, current_time)
                    detection_stats['face_detections'] = len(recognition_results)

            elif detection_strategy == "body_only":
                # Distant people: Use body detection only
                recognition_results = self._process_body_recognition(bodies, frame, current_time)
                detection_stats['body_detections'] = len(recognition_results)

            elif detection_strategy == "face_body_fusion":
                # Mixed scenario: Use both face and body with intelligent fusion
                recognition_results = self._process_fusion_recognition(faces, bodies, frame, current_time)
                detection_stats['fusion_detections'] = len(recognition_results)

            elif detection_strategy == "body_fallback":
                # Face detection failed: Fallback to body detection
                recognition_results = self._process_body_recognition(bodies, frame, current_time)
                detection_stats['body_detections'] = len(recognition_results)

            detection_stats['total_detections'] = len(recognition_results)
            detection_stats['recognition_results'] = recognition_results

            logger.debug(f"🧠 INTELLIGENT DETECTION: Strategy={detection_strategy}, Total={detection_stats['total_detections']}")

            return detection_stats

        except Exception as e:
            logger.error(f"Error in intelligent detection coordinator: {e}")
            return {
                'recognition_results': [],
                'detection_method': 'error',
                'total_detections': 0,
                'face_detections': 0,
                'body_detections': 0,
                'fusion_detections': 0
            }

    def _analyze_detection_strategy(self, faces: List[Dict], bodies: List[Dict], frame) -> str:
        """INTELLIGENT ANALYSIS: Determine the best detection strategy based on current conditions"""
        try:
            frame_height, frame_width = frame.shape[:2]

            # Analyze face detection quality
            face_analysis = self._analyze_face_detection_quality(faces, frame_width, frame_height)
            body_analysis = self._analyze_body_detection_quality(bodies, frame_width, frame_height)

            # Decision logic based on analysis
            if face_analysis['has_good_faces'] and not face_analysis['has_tiny_faces']:
                # Good face detection available, no tiny faces
                return "face_only"

            elif face_analysis['has_tiny_faces'] and body_analysis['has_bodies']:
                # Tiny faces detected but bodies available - use fusion
                return "face_body_fusion"

            elif not face_analysis['has_faces'] and body_analysis['has_bodies']:
                # No faces but bodies detected - use body only
                return "body_only"

            elif face_analysis['has_faces'] and body_analysis['has_bodies']:
                # Both available - use fusion for maximum accuracy
                return "face_body_fusion"

            elif not face_analysis['has_faces'] and not body_analysis['has_bodies']:
                # Nothing detected - try body fallback
                return "body_fallback"

            else:
                # Default to face detection
                return "face_only"

        except Exception as e:
            logger.error(f"Error analyzing detection strategy: {e}")
            return "face_only"

    def _analyze_face_detection_quality(self, faces: List[Dict], frame_width: int, frame_height: int) -> Dict[str, Any]:
        """Analyze face detection quality to determine strategy"""
        try:
            analysis = {
                'has_faces': len(faces) > 0,
                'has_good_faces': False,
                'has_tiny_faces': False,
                'average_face_size': 0,
                'face_count': len(faces),
                'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
            }

            if not faces:
                return analysis

            total_size = 0
            for face in faces:
                bbox = face.get('bbox', [])
                if len(bbox) >= 4:
                    width, height = bbox[2], bbox[3]
                    face_size = min(width, height)
                    total_size += face_size

                    # Check for tiny faces
                    if face_size <= 30:
                        analysis['has_tiny_faces'] = True

                    # Check for good faces
                    if face_size >= 50 and face.get('confidence', 0) >= 0.5:
                        analysis['has_good_faces'] = True

                    # Quality distribution
                    quality = face.get('face_quality', 'unknown')
                    if quality in analysis['quality_distribution']:
                        analysis['quality_distribution'][quality] += 1

            analysis['average_face_size'] = total_size / len(faces) if faces else 0

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing face detection quality: {e}")
            return {'has_faces': False, 'has_good_faces': False, 'has_tiny_faces': False}

    def _analyze_body_detection_quality(self, bodies: List[Dict], frame_width: int, frame_height: int) -> Dict[str, Any]:
        """Analyze body detection quality"""
        try:
            analysis = {
                'has_bodies': len(bodies) > 0,
                'has_good_bodies': False,
                'body_count': len(bodies),
                'average_body_size': 0
            }

            if not bodies:
                return analysis

            total_size = 0
            for body in bodies:
                bbox = body.get('bbox', [])
                if len(bbox) >= 4:
                    width, height = bbox[2], bbox[3]
                    body_size = width * height
                    total_size += body_size

                    # Check for good bodies (reasonable size and confidence)
                    if body_size >= 5000 and body.get('confidence', 0) >= self.body_confidence_threshold:
                        analysis['has_good_bodies'] = True

            analysis['average_body_size'] = total_size / len(bodies) if bodies else 0

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing body detection quality: {e}")
            return {'has_bodies': False, 'has_good_bodies': False}

    def _generate_body_detection_from_faces(self, faces: List[Dict], frame) -> List[Dict]:
        """Generate body detection from face detection for distant people"""
        try:
            bodies = []
            frame_height, frame_width = frame.shape[:2]

            for face in faces:
                bbox = face.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox
                    face_size = min(w, h)

                    # For small faces, estimate body position
                    if face_size <= self.face_size_threshold_for_body:
                        # Estimate body dimensions based on face size
                        body_width = w * 3  # Body is roughly 3x face width
                        body_height = h * 6  # Body is roughly 6x face height

                        # Position body below face
                        body_x = max(0, x - body_width // 4)  # Center body on face
                        body_y = y  # Start from face position

                        # Ensure body fits in frame
                        body_width = min(body_width, frame_width - body_x)
                        body_height = min(body_height, frame_height - body_y)

                        if body_width > 20 and body_height > 40:  # Reasonable body size
                            body_data = {
                                'bbox': [body_x, body_y, body_width, body_height],
                                'confidence': face.get('confidence', 0.5) * 0.8,  # Slightly lower confidence
                                'source': 'face_estimation',
                                'face_reference': face,
                                'estimated': True
                            }
                            bodies.append(body_data)
                            logger.debug(f"🔍 ESTIMATED BODY from tiny face: {face_size}px -> {body_width}x{body_height}")

            return bodies

        except Exception as e:
            logger.error(f"Error generating body detection from faces: {e}")
            return []

    def _process_body_recognition(self, bodies: List[Dict], frame, current_time: float) -> List[Dict]:
        """Process body-only recognition for distant people"""
        try:
            recognition_results = []

            for body in bodies:
                bbox = body.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox
                    confidence = body.get('confidence', 0.5)

                    # Calculate body-based parameters
                    body_area = w * h
                    body_center = [x + w//2, y + h//2]

                    # Generate unknown person ID for body detection
                    unknown_id = f"BODY_UNKNOWN_{self.next_unknown_id:03d}"
                    self.next_unknown_id += 1

                    # Calculate engagement parameters based on body characteristics
                    engagement_score = self._calculate_body_engagement(body, bbox, confidence)
                    attention_score = self._calculate_body_attention(body, bbox)
                    participation_score = self._calculate_body_participation(body, bbox, confidence)

                    recognition_result = {
                        'face_bbox': bbox,  # Using body bbox as reference
                        'roll_number': unknown_id,
                        'name': f"Distant Person {self.next_unknown_id-1}",
                        'confidence': confidence,
                        'face_encoding': None,  # No face encoding for body-only detection
                        'is_unknown': True,
                        'detection_method': 'body_only',
                        'engagement_score': engagement_score,
                        'attention_score': attention_score,
                        'participation_score': participation_score,
                        'overall_score': (engagement_score + attention_score + participation_score) / 3.0,
                        'body_area': body_area,
                        'body_center': body_center,
                        'estimated_from_face': body.get('estimated', False)
                    }

                    recognition_results.append(recognition_result)
                    logger.info(f"👤 BODY DETECTION: {unknown_id} - Body Area: {body_area}, Engagement: {engagement_score:.1f}")

            return recognition_results

        except Exception as e:
            logger.error(f"Error in body recognition: {e}")
            return []

    def _process_fusion_recognition(self, faces: List[Dict], bodies: List[Dict], frame, current_time: float) -> List[Dict]:
        """Process fusion recognition combining face and body detection"""
        try:
            recognition_results = []

            # First, process face recognition normally
            face_results = []
            if faces and self.frame_count % self.face_recognition_interval == 0:
                face_results = self._recognize_faces(frame, faces, current_time)

            # Then, process body detection
            body_results = self._process_body_recognition(bodies, frame, current_time)

            # Fusion: Associate faces with bodies when possible
            fused_results = self._associate_faces_with_bodies(face_results, body_results)

            # Add unmatched face results
            for face_result in face_results:
                if not any(f.get('face_id') == face_result.get('roll_number') for f in fused_results):
                    recognition_results.append(face_result)

            # Add unmatched body results
            for body_result in body_results:
                if not any(f.get('body_id') == body_result.get('roll_number') for f in fused_results):
                    recognition_results.append(body_result)

            # Add fused results
            recognition_results.extend(fused_results)

            logger.debug(f"🔗 FUSION DETECTION: {len(face_results)} faces + {len(body_results)} bodies = {len(recognition_results)} total")

            return recognition_results

        except Exception as e:
            logger.error(f"Error in fusion recognition: {e}")
            return []

    def _calculate_body_engagement(self, body: Dict, bbox: List, confidence: float) -> float:
        """Calculate engagement score based on body characteristics"""
        try:
            base_score = min(confidence * 100, 80)  # Base score from detection confidence

            # Adjust based on body size (larger bodies may indicate closer/more engaged people)
            body_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            size_multiplier = min(1.0, body_area / 20000.0) * 0.3 + 0.7  # 0.7 to 1.0 range

            # Adjust based on body position (center of frame = higher engagement)
            if len(bbox) >= 4:
                body_center_x = bbox[0] + bbox[2] / 2
                frame_center_x = 320  # Assume frame width/2
                distance_from_center = abs(body_center_x - frame_center_x) / frame_center_x
                position_multiplier = max(0.8, 1.0 - distance_from_center * 0.2)
            else:
                position_multiplier = 0.9

            engagement_score = base_score * size_multiplier * position_multiplier
            return max(0, min(100, engagement_score))

        except Exception as e:
            logger.error(f"Error calculating body engagement: {e}")
            return 60.0

    def _calculate_body_attention(self, body: Dict, bbox: List) -> float:
        """Calculate attention score based on body position and posture"""
        try:
            base_attention = 65.0  # Base attention for body detection

            # Adjust based on body position in frame
            if len(bbox) >= 4:
                body_center_x = bbox[0] + bbox[2] / 2
                body_center_y = bbox[1] + bbox[3] / 2

                # Assume frame dimensions
                frame_center_x = 320
                frame_center_y = 240

                # Distance from center (normalized)
                distance_from_center = ((body_center_x - frame_center_x)**2 + (body_center_y - frame_center_y)**2)**0.5
                max_distance = (frame_center_x**2 + frame_center_y**2)**0.5
                center_score = max(0, 1.0 - (distance_from_center / max_distance)) * 25

                attention_score = base_attention + center_score
            else:
                attention_score = base_attention

            return max(0, min(100, attention_score))

        except Exception as e:
            logger.error(f"Error calculating body attention: {e}")
            return 65.0

    def _calculate_body_participation(self, body: Dict, bbox: List, confidence: float) -> float:
        """Calculate participation score based on body size and confidence"""
        try:
            base_participation = confidence * 70

            # Adjust based on body size (larger bodies may indicate active participation)
            body_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            size_bonus = min(25, body_area / 2000.0)  # Up to 25 points bonus

            participation_score = base_participation + size_bonus
            return max(0, min(100, participation_score))

        except Exception as e:
            logger.error(f"Error calculating body participation: {e}")
            return 55.0

    def _associate_faces_with_bodies(self, face_results: List[Dict], body_results: List[Dict]) -> List[Dict]:
        """Associate face detections with body detections for fusion"""
        try:
            fused_results = []

            for face_result in face_results:
                face_bbox = face_result.get('face_bbox', [])
                if len(face_bbox) < 4:
                    continue

                face_center = [face_bbox[0] + face_bbox[2]//2, face_bbox[1] + face_bbox[3]//2]

                # Find closest body
                closest_body = None
                min_distance = float('inf')

                for body_result in body_results:
                    body_bbox = body_result.get('face_bbox', [])  # Using face_bbox field for body
                    if len(body_bbox) < 4:
                        continue

                    body_center = body_result.get('body_center', [body_bbox[0] + body_bbox[2]//2, body_bbox[1] + body_bbox[3]//2])

                    # Calculate distance between face and body centers
                    distance = ((face_center[0] - body_center[0])**2 + (face_center[1] - body_center[1])**2)**0.5

                    if distance < min_distance and distance < self.fusion_distance_threshold:
                        min_distance = distance
                        closest_body = body_result

                # Create fused result if association found
                if closest_body:
                    fused_result = face_result.copy()
                    fused_result.update({
                        'detection_method': 'face_body_fusion',
                        'body_data': closest_body,
                        'body_engagement': closest_body.get('engagement_score', 0),
                        'body_attention': closest_body.get('attention_score', 0),
                        'body_participation': closest_body.get('participation_score', 0),
                        'fusion_distance': min_distance,
                        'face_id': face_result.get('roll_number'),
                        'body_id': closest_body.get('roll_number')
                    })

                    # Average the scores for better accuracy
                    fused_result['engagement_score'] = (face_result.get('engagement_score', 0) + closest_body.get('engagement_score', 0)) / 2
                    fused_result['attention_score'] = (face_result.get('attention_score', 0) + closest_body.get('attention_score', 0)) / 2
                    fused_result['participation_score'] = (face_result.get('participation_score', 0) + closest_body.get('participation_score', 0)) / 2
                    fused_result['overall_score'] = (fused_result['engagement_score'] + fused_result['attention_score'] + fused_result['participation_score']) / 3

                    fused_results.append(fused_result)
                    logger.debug(f"🔗 FUSION: Associated {face_result.get('name', 'Unknown')} with body (distance: {min_distance:.1f})")

            return fused_results

        except Exception as e:
            logger.error(f"Error associating faces with bodies: {e}")
            return []

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            'tracked_persons': [],
            'attendance_count': 0,
            'total_recognized': 0,
            'active_alerts': [],
            'frame_annotations': []
        }

    def add_detection_feedback(self, feedback_type: str, detection_data: Dict[str, Any], user_correction: Dict[str, Any] = None):
        """Add feedback for detection performance to improve future detections"""
        try:
            current_time = time.time()

            feedback_instance = {
                'timestamp': current_time,
                'feedback_type': feedback_type,  # 'false_positive', 'false_negative', 'correct_detection', 'missed_person'
                'detection_data': detection_data,
                'user_correction': user_correction,
                'session_id': f"session_{int(self.session_start_time)}",
                'frame_count': self.frame_count,
                'current_thresholds': {
                    'face_threshold': self.adaptive_face_threshold,
                    'body_threshold': self.adaptive_body_threshold,
                    'recognition_threshold': self.adaptive_recognition_threshold
                }
            }

            # Add to appropriate buffer
            self.feedback_buffer.append(feedback_instance)

            if feedback_type in ['false_positive', 'false_negative', 'missed_person']:
                self.negative_feedback_buffer.append(feedback_instance)
                logger.info(f"📉 NEGATIVE FEEDBACK: {feedback_type} - Will adjust thresholds")
            else:
                self.positive_feedback_buffer.append(feedback_instance)
                logger.debug(f"📈 POSITIVE FEEDBACK: {feedback_type}")

            # Process feedback immediately for real-time learning
            self._process_feedback_immediately(feedback_instance)

            # Update performance metrics
            self._update_detection_performance_metrics(feedback_type)

            # Save feedback to checkpoint
            if len(self.feedback_buffer) % 10 == 0:  # Save every 10 feedback instances
                self._save_learning_checkpoint()

        except Exception as e:
            logger.error(f"Error adding detection feedback: {e}")

    def _process_feedback_immediately(self, feedback_instance: Dict[str, Any]):
        """Process feedback immediately for real-time threshold adjustment"""
        try:
            feedback_type = feedback_instance['feedback_type']
            detection_data = feedback_instance['detection_data']

            # NEGATIVE FEEDBACK PROCESSING
            if feedback_type == 'false_positive':
                # Detection was wrong - increase thresholds to be more strict
                self._adjust_thresholds_for_false_positive(detection_data)

            elif feedback_type == 'false_negative' or feedback_type == 'missed_person':
                # Missed a person - decrease thresholds to be more sensitive
                self._adjust_thresholds_for_false_negative(detection_data)

            elif feedback_type == 'incorrect_recognition':
                # Wrong person identified - adjust recognition threshold
                self._adjust_recognition_threshold_for_error(detection_data)

            # POSITIVE FEEDBACK PROCESSING
            elif feedback_type == 'correct_detection':
                # Good detection - slightly reinforce current thresholds
                self._reinforce_current_thresholds(detection_data)

            elif feedback_type == 'correct_recognition':
                # Good recognition - reinforce recognition threshold
                self._reinforce_recognition_threshold(detection_data)

            logger.debug(f"🔄 THRESHOLD UPDATE: Face={self.adaptive_face_threshold:.3f}, Body={self.adaptive_body_threshold:.3f}, Recognition={self.adaptive_recognition_threshold:.3f}")

        except Exception as e:
            logger.error(f"Error processing feedback immediately: {e}")

    def _adjust_thresholds_for_false_positive(self, detection_data: Dict[str, Any]):
        """Adjust thresholds when false positive occurs"""
        try:
            detection_method = detection_data.get('detection_method', 'unknown')
            confidence = detection_data.get('confidence', 0.5)

            # Increase thresholds to be more strict
            if 'face' in detection_method:
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative
                self.adaptive_face_threshold = min(0.8, self.adaptive_face_threshold + adjustment)

            if 'body' in detection_method:
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative
                self.adaptive_body_threshold = min(0.8, self.adaptive_body_threshold + adjustment)

            if 'recognition' in detection_method:
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative
                self.adaptive_recognition_threshold = min(0.9, self.adaptive_recognition_threshold + adjustment)

            logger.info(f"🔺 FALSE POSITIVE ADJUSTMENT: Increased thresholds for {detection_method}")

        except Exception as e:
            logger.error(f"Error adjusting thresholds for false positive: {e}")

    def _adjust_thresholds_for_false_negative(self, detection_data: Dict[str, Any]):
        """Adjust thresholds when false negative occurs (missed detection)"""
        try:
            detection_method = detection_data.get('detection_method', 'unknown')

            # Decrease thresholds to be more sensitive
            if 'face' in detection_method or detection_method == 'unknown':
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative
                self.adaptive_face_threshold = max(0.01, self.adaptive_face_threshold - adjustment)

            if 'body' in detection_method or detection_method == 'unknown':
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative
                self.adaptive_body_threshold = max(0.1, self.adaptive_body_threshold - adjustment)

            logger.info(f"🔻 FALSE NEGATIVE ADJUSTMENT: Decreased thresholds for {detection_method}")

        except Exception as e:
            logger.error(f"Error adjusting thresholds for false negative: {e}")

    def _adjust_recognition_threshold_for_error(self, detection_data: Dict[str, Any]):
        """Adjust recognition threshold when wrong person is identified"""
        try:
            confidence = detection_data.get('confidence', 0.5)

            # If confidence was high but recognition was wrong, increase threshold
            if confidence > 0.7:
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative * 1.5
                self.adaptive_recognition_threshold = min(0.9, self.adaptive_recognition_threshold + adjustment)
            else:
                # If confidence was low, moderate adjustment
                adjustment = self.threshold_adjustment_factor * self.feedback_weight_negative
                self.adaptive_recognition_threshold = min(0.9, self.adaptive_recognition_threshold + adjustment)

            logger.info(f"🔺 RECOGNITION ERROR ADJUSTMENT: Increased recognition threshold to {self.adaptive_recognition_threshold:.3f}")

        except Exception as e:
            logger.error(f"Error adjusting recognition threshold for error: {e}")

    def _reinforce_current_thresholds(self, detection_data: Dict[str, Any]):
        """Reinforce current thresholds when detection is correct"""
        try:
            detection_method = detection_data.get('detection_method', 'unknown')

            # Small positive reinforcement (move slightly towards current values)
            reinforcement = self.threshold_adjustment_factor * self.feedback_weight_positive * 0.5

            if 'face' in detection_method:
                # Move slightly towards the confidence that worked
                target_confidence = detection_data.get('confidence', self.adaptive_face_threshold)
                self.adaptive_face_threshold = self.adaptive_face_threshold * 0.95 + target_confidence * 0.05

            if 'body' in detection_method:
                target_confidence = detection_data.get('confidence', self.adaptive_body_threshold)
                self.adaptive_body_threshold = self.adaptive_body_threshold * 0.95 + target_confidence * 0.05

            logger.debug(f"✅ POSITIVE REINFORCEMENT: Reinforced thresholds for {detection_method}")

        except Exception as e:
            logger.error(f"Error reinforcing current thresholds: {e}")

    def _reinforce_recognition_threshold(self, detection_data: Dict[str, Any]):
        """Reinforce recognition threshold when recognition is correct"""
        try:
            confidence = detection_data.get('confidence', self.adaptive_recognition_threshold)

            # Move recognition threshold slightly towards the confidence that worked
            self.adaptive_recognition_threshold = self.adaptive_recognition_threshold * 0.98 + confidence * 0.02

            logger.debug(f"✅ RECOGNITION REINFORCEMENT: Adjusted to {self.adaptive_recognition_threshold:.3f}")

        except Exception as e:
            logger.error(f"Error reinforcing recognition threshold: {e}")

    def _update_detection_performance_metrics(self, feedback_type: str):
        """Update performance metrics based on feedback"""
        try:
            current_time = time.time()

            performance_data = {
                'timestamp': current_time,
                'feedback_type': feedback_type,
                'thresholds': {
                    'face': self.adaptive_face_threshold,
                    'body': self.adaptive_body_threshold,
                    'recognition': self.adaptive_recognition_threshold
                },
                'session_duration': current_time - self.session_start_time
            }

            self.detection_performance_history.append(performance_data)

            # Calculate recent performance metrics
            if len(self.detection_performance_history) >= self.performance_window_size:
                recent_feedback = list(self.detection_performance_history)[-self.performance_window_size:]

                positive_count = sum(1 for f in recent_feedback if f['feedback_type'] in ['correct_detection', 'correct_recognition'])
                negative_count = sum(1 for f in recent_feedback if f['feedback_type'] in ['false_positive', 'false_negative', 'missed_person', 'incorrect_recognition'])

                if positive_count + negative_count > 0:
                    accuracy = positive_count / (positive_count + negative_count)
                    logger.info(f"📊 RECENT PERFORMANCE: {accuracy:.2%} accuracy over last {self.performance_window_size} feedback instances")

                    # Auto-adjust learning rate based on performance
                    if accuracy < 0.7:  # Poor performance
                        self.learning_rate_detection = min(0.05, self.learning_rate_detection * 1.1)
                        logger.info(f"📈 INCREASED LEARNING RATE: {self.learning_rate_detection:.3f} (poor performance)")
                    elif accuracy > 0.9:  # Excellent performance
                        self.learning_rate_detection = max(0.005, self.learning_rate_detection * 0.9)
                        logger.info(f"📉 DECREASED LEARNING RATE: {self.learning_rate_detection:.3f} (excellent performance)")

        except Exception as e:
            logger.error(f"Error updating detection performance metrics: {e}")

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback and learning performance"""
        try:
            total_feedback = len(self.feedback_buffer)
            positive_feedback = len(self.positive_feedback_buffer)
            negative_feedback = len(self.negative_feedback_buffer)

            recent_performance = []
            if len(self.detection_performance_history) >= 10:
                recent_feedback = list(self.detection_performance_history)[-10:]
                positive_recent = sum(1 for f in recent_feedback if f['feedback_type'] in ['correct_detection', 'correct_recognition'])
                total_recent = len(recent_feedback)
                recent_accuracy = positive_recent / total_recent if total_recent > 0 else 0
                recent_performance = [recent_accuracy]

            return {
                'total_feedback_instances': total_feedback,
                'positive_feedback_count': positive_feedback,
                'negative_feedback_count': negative_feedback,
                'feedback_ratio': positive_feedback / max(1, total_feedback),
                'current_thresholds': {
                    'face_threshold': self.adaptive_face_threshold,
                    'body_threshold': self.adaptive_body_threshold,
                    'recognition_threshold': self.adaptive_recognition_threshold
                },
                'learning_rate': self.learning_rate_detection,
                'recent_accuracy': recent_performance[0] if recent_performance else 0,
                'session_duration': time.time() - self.session_start_time,
                'total_detections': self.total_detections
            }

        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            return {}
