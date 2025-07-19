"""
Continuous Learning System for Real-time Model Improvement
Implements reinforcement learning, feedback loops, and active learning
to continuously improve model precision, mAP, accuracy, and confidence
"""

import numpy as np
import pandas as pd
import json
import time
import threading
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib

from src.utils.base_processor import BaseProcessor
from src.utils.logger import logger

class FeedbackType(Enum):
    """Types of feedback for model improvement"""
    CORRECT_PREDICTION = "correct"
    INCORRECT_PREDICTION = "incorrect"
    TEACHER_FEEDBACK = "teacher"
    SYSTEM_VALIDATION = "system"
    UNCERTAINTY_SAMPLING = "uncertainty"

@dataclass
class LearningInstance:
    """Single learning instance with features and labels"""
    timestamp: float
    features: Dict[str, Any]
    predicted_label: str
    actual_label: Optional[str]
    confidence: float
    feedback_type: FeedbackType
    model_version: str
    session_id: str
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_avg: float
    sample_count: int
    timestamp: float
    model_version: str

class ContinuousLearningSystem(BaseProcessor):
    """Continuous learning system for real-time model improvement"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ContinuousLearningSystem", config)
        
        # Learning configuration
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 32)
        self.validation_split = config.get('validation_split', 0.2)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.7)
        
        # Data management
        self.data_dir = Path("data")
        self.models_dir = self.data_dir / "models"
        self.datasets_dir = self.data_dir / "datasets"
        self.feedback_dir = self.data_dir / "feedback"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.datasets_dir, self.feedback_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Learning components
        self.feedback_collector = FeedbackCollector()
        self.active_learner = ActiveLearner()
        self.model_validator = ModelValidator()
        self.dataset_manager = DatasetManager(self.datasets_dir)
        self.performance_tracker = PerformanceTracker()
        
        # Learning data
        self.learning_instances = deque(maxlen=10000)
        self.feedback_queue = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        
        # Model references (will be injected)
        self.models = {}
        self.model_versions = {}
        
        # Learning thread
        self.learning_thread = None
        self.learning_active = False
        
        # Session tracking
        self.current_session_id = f"session_{int(time.time())}"
        
    def initialize(self) -> bool:
        """Initialize continuous learning system"""
        try:
            logger.info("Initializing continuous learning system...")
            
            # Initialize components
            self.feedback_collector.initialize()
            self.active_learner.initialize()
            self.model_validator.initialize()
            self.dataset_manager.initialize()
            self.performance_tracker.initialize()
            
            # Load existing datasets
            self._load_initial_datasets()
            
            # Start learning thread
            self._start_learning_thread()
            
            logger.info("Continuous learning system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize continuous learning system: {e}")
            return False
    
    def register_model(self, model_name: str, model_instance: Any):
        """Register a model for continuous learning"""
        self.models[model_name] = model_instance
        self.model_versions[model_name] = "v1.0"
        logger.info(f"Registered model: {model_name}")
    
    def add_prediction_feedback(self, model_name: str, features: Dict[str, Any], 
                              predicted_label: str, confidence: float,
                              actual_label: Optional[str] = None,
                              feedback_type: FeedbackType = FeedbackType.SYSTEM_VALIDATION):
        """Add prediction feedback for learning"""
        try:
            learning_instance = LearningInstance(
                timestamp=time.time(),
                features=features,
                predicted_label=predicted_label,
                actual_label=actual_label,
                confidence=confidence,
                feedback_type=feedback_type,
                model_version=self.model_versions.get(model_name, "v1.0"),
                session_id=self.current_session_id,
                metadata={'model_name': model_name}
            )
            
            self.learning_instances.append(learning_instance)
            
            # Add to feedback queue for immediate processing
            if actual_label is not None:
                self.feedback_queue.append(learning_instance)
            
            # Check for active learning opportunities
            if confidence < self.uncertainty_threshold:
                self.active_learner.add_uncertain_sample(learning_instance)
            
        except Exception as e:
            logger.error(f"Error adding prediction feedback: {e}")
    
    def add_teacher_feedback(self, model_name: str, features: Dict[str, Any],
                           predicted_label: str, correct_label: str, confidence: float):
        """Add teacher feedback for supervised learning"""
        self.add_prediction_feedback(
            model_name=model_name,
            features=features,
            predicted_label=predicted_label,
            confidence=confidence,
            actual_label=correct_label,
            feedback_type=FeedbackType.TEACHER_FEEDBACK
        )
        logger.info(f"Teacher feedback added for {model_name}: {predicted_label} -> {correct_label}")
    
    def get_performance_metrics(self, model_name: str) -> Dict[str, float]:
        """Get current performance metrics for a model"""
        return self.performance_tracker.get_metrics(model_name)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            'total_instances': len(self.learning_instances),
            'feedback_queue_size': len(self.feedback_queue),
            'active_learning_samples': self.active_learner.get_sample_count(),
            'model_versions': self.model_versions.copy(),
            'performance_history': list(self.performance_history)[-10:],  # Last 10 entries
            'current_session': self.current_session_id,
            'learning_active': self.learning_active
        }
    
    def _load_initial_datasets(self):
        """Load initial datasets for bootstrapping"""
        try:
            # Load engagement datasets
            engagement_dataset = self.dataset_manager.load_engagement_dataset()
            if engagement_dataset:
                logger.info(f"Loaded engagement dataset with {len(engagement_dataset)} samples")
                self._bootstrap_models_with_dataset(engagement_dataset)
            else:
                logger.info("No initial engagement dataset found, will learn from scratch")
            
            # Load behavioral datasets
            behavioral_dataset = self.dataset_manager.load_behavioral_dataset()
            if behavioral_dataset:
                logger.info(f"Loaded behavioral dataset with {len(behavioral_dataset)} samples")
            
        except Exception as e:
            logger.error(f"Error loading initial datasets: {e}")
    
    def _bootstrap_models_with_dataset(self, dataset: List[Dict[str, Any]]):
        """Bootstrap models with initial dataset"""
        try:
            if len(dataset) < 10:
                logger.warning("Dataset too small for bootstrapping")
                return
            
            # Convert to learning instances
            for data in dataset:
                learning_instance = LearningInstance(
                    timestamp=time.time(),
                    features=data.get('features', {}),
                    predicted_label=data.get('label', 'unknown'),
                    actual_label=data.get('label', 'unknown'),
                    confidence=1.0,  # Assume ground truth
                    feedback_type=FeedbackType.SYSTEM_VALIDATION,
                    model_version="bootstrap",
                    session_id="bootstrap_session",
                    metadata=data.get('metadata', {})
                )
                self.learning_instances.append(learning_instance)
            
            logger.info(f"Bootstrapped with {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Error bootstrapping models: {e}")
    
    def _start_learning_thread(self):
        """Start background learning thread"""
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("Learning thread started")
    
    def _learning_loop(self):
        """Main learning loop running in background"""
        while self.learning_active:
            try:
                # Process feedback queue
                if len(self.feedback_queue) >= self.batch_size:
                    self._process_feedback_batch()
                
                # Periodic model retraining
                if len(self.learning_instances) % 100 == 0 and len(self.learning_instances) > 0:
                    self._retrain_models()
                
                # Validate model performance
                if len(self.learning_instances) % 50 == 0 and len(self.learning_instances) > 0:
                    self._validate_model_performance()
                
                # Active learning
                self.active_learner.process_uncertain_samples()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _process_feedback_batch(self):
        """Process a batch of feedback for immediate learning"""
        try:
            batch = []
            for _ in range(min(self.batch_size, len(self.feedback_queue))):
                if self.feedback_queue:
                    batch.append(self.feedback_queue.popleft())
            
            if not batch:
                return
            
            # Group by model
            model_batches = {}
            for instance in batch:
                model_name = instance.metadata.get('model_name', 'unknown')
                if model_name not in model_batches:
                    model_batches[model_name] = []
                model_batches[model_name].append(instance)
            
            # Update each model
            for model_name, instances in model_batches.items():
                if model_name in self.models:
                    self._update_model_with_feedback(model_name, instances)
            
            logger.info(f"Processed feedback batch of {len(batch)} instances")
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
    
    def _update_model_with_feedback(self, model_name: str, instances: List[LearningInstance]):
        """Update a specific model with feedback instances"""
        try:
            model = self.models[model_name]
            
            # Prepare training data
            X = []
            y = []
            
            for instance in instances:
                if instance.actual_label is not None:
                    feature_vector = self._extract_feature_vector(instance.features)
                    if feature_vector:
                        X.append(feature_vector)
                        y.append(instance.actual_label)
            
            if len(X) < 5:  # Need minimum samples
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Incremental learning (if supported)
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X, y)
            else:
                # Full retraining with recent data
                recent_instances = list(self.learning_instances)[-1000:]  # Last 1000 instances
                X_full, y_full = self._prepare_training_data(recent_instances, model_name)
                
                if len(X_full) > 10:
                    model.fit(X_full, y_full)
            
            # Update model version
            current_version = self.model_versions.get(model_name, "v1.0")
            version_num = float(current_version.replace('v', '')) + 0.1
            self.model_versions[model_name] = f"v{version_num:.1f}"
            
            logger.info(f"Updated model {model_name} to version {self.model_versions[model_name]}")
            
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {e}")
    
    def _retrain_models(self):
        """Retrain all models with accumulated data"""
        try:
            logger.info("Starting model retraining...")
            
            for model_name, model in self.models.items():
                # Prepare training data
                X, y = self._prepare_training_data(list(self.learning_instances), model_name)
                
                if len(X) > 20:  # Minimum samples for retraining
                    # Split for validation
                    split_idx = int(len(X) * (1 - self.validation_split))
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # Retrain model
                    model.fit(X_train, y_train)
                    
                    # Validate performance
                    if len(X_val) > 0:
                        y_pred = model.predict(X_val)
                        accuracy = accuracy_score(y_val, y_pred)
                        
                        # Update performance tracking
                        performance = ModelPerformance(
                            accuracy=accuracy,
                            precision=precision_score(y_val, y_pred, average='weighted', zero_division=0),
                            recall=recall_score(y_val, y_pred, average='weighted', zero_division=0),
                            f1_score=f1_score(y_val, y_pred, average='weighted', zero_division=0),
                            confidence_avg=0.8,  # Placeholder
                            sample_count=len(X),
                            timestamp=time.time(),
                            model_version=self.model_versions.get(model_name, "v1.0")
                        )
                        
                        self.performance_tracker.add_performance(model_name, performance)
                        self.performance_history.append(asdict(performance))
                        
                        logger.info(f"Retrained {model_name}: Accuracy={accuracy:.3f}, Samples={len(X)}")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _validate_model_performance(self):
        """Validate current model performance"""
        try:
            for model_name, model in self.models.items():
                # Get recent predictions for validation
                recent_instances = [inst for inst in list(self.learning_instances)[-100:] 
                                 if inst.metadata.get('model_name') == model_name and inst.actual_label is not None]
                
                if len(recent_instances) > 10:
                    X, y_true = self._prepare_validation_data(recent_instances)
                    
                    if len(X) > 0:
                        y_pred = model.predict(X)
                        accuracy = accuracy_score(y_true, y_pred)
                        
                        # Check for performance degradation
                        previous_performance = self.performance_tracker.get_latest_performance(model_name)
                        if previous_performance and accuracy < previous_performance.accuracy * 0.9:
                            logger.warning(f"Performance degradation detected for {model_name}: "
                                         f"{accuracy:.3f} < {previous_performance.accuracy:.3f}")
                            # Trigger retraining
                            self._retrain_models()
            
        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
    
    def _prepare_training_data(self, instances: List[LearningInstance], model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from learning instances"""
        X = []
        y = []
        
        for instance in instances:
            if (instance.metadata.get('model_name') == model_name and 
                instance.actual_label is not None):
                
                feature_vector = self._extract_feature_vector(instance.features)
                if feature_vector:
                    X.append(feature_vector)
                    y.append(instance.actual_label)
        
        return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])
    
    def _prepare_validation_data(self, instances: List[LearningInstance]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare validation data from learning instances"""
        X = []
        y = []
        
        for instance in instances:
            feature_vector = self._extract_feature_vector(instance.features)
            if feature_vector and instance.actual_label is not None:
                X.append(feature_vector)
                y.append(instance.actual_label)
        
        return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])
    
    def _extract_feature_vector(self, features: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from features dictionary"""
        try:
            # This should be customized based on your feature structure
            feature_vector = []
            
            # Extract numerical features
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, dict):
                    # Recursively extract from nested dictionaries
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (int, float)):
                            feature_vector.append(float(nested_value))
            
            return feature_vector if len(feature_vector) > 0 else None
            
        except Exception as e:
            logger.error(f"Error extracting feature vector: {e}")
            return None
    
    def _save_models(self):
        """Save all models to disk"""
        try:
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{model_name}_{self.model_versions[model_name]}.pkl"
                joblib.dump(model, model_path)
            
            # Save learning instances for future use
            instances_path = self.datasets_dir / f"learning_instances_{self.current_session_id}.pkl"
            with open(instances_path, 'wb') as f:
                pickle.dump(list(self.learning_instances), f)
            
            logger.info("Models and learning data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def cleanup(self):
        """Cleanup continuous learning system"""
        self.learning_active = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
        
        # Final save
        self._save_models()
        
        logger.info("Continuous learning system cleaned up")

# Helper classes for learning components
class FeedbackCollector:
    """Collect and manage feedback from various sources"""
    
    def initialize(self):
        logger.info("Feedback collector initialized")

class ActiveLearner:
    """Active learning for uncertain samples"""
    
    def __init__(self):
        self.uncertain_samples = deque(maxlen=1000)
    
    def initialize(self):
        logger.info("Active learner initialized")
    
    def add_uncertain_sample(self, instance: LearningInstance):
        """Add uncertain sample for active learning"""
        self.uncertain_samples.append(instance)
    
    def get_sample_count(self) -> int:
        """Get number of uncertain samples"""
        return len(self.uncertain_samples)
    
    def process_uncertain_samples(self):
        """Process uncertain samples for active learning"""
        # Placeholder for active learning logic
        pass

class ModelValidator:
    """Validate model performance and detect issues"""
    
    def initialize(self):
        logger.info("Model validator initialized")

class DatasetManager:
    """Manage datasets and data loading"""
    
    def __init__(self, datasets_dir: Path):
        self.datasets_dir = datasets_dir
    
    def initialize(self):
        logger.info("Dataset manager initialized")
    
    def load_engagement_dataset(self) -> Optional[List[Dict[str, Any]]]:
        """Load engagement dataset"""
        # Placeholder - would load from file or database
        return None
    
    def load_behavioral_dataset(self) -> Optional[List[Dict[str, Any]]]:
        """Load behavioral dataset"""
        # Placeholder - would load from file or database
        return None

class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.performance_history = {}
    
    def initialize(self):
        logger.info("Performance tracker initialized")
    
    def add_performance(self, model_name: str, performance: ModelPerformance):
        """Add performance metrics for a model"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = deque(maxlen=100)
        self.performance_history[model_name].append(performance)
    
    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """Get latest metrics for a model"""
        if model_name in self.performance_history and self.performance_history[model_name]:
            latest = self.performance_history[model_name][-1]
            return {
                'accuracy': latest.accuracy,
                'precision': latest.precision,
                'recall': latest.recall,
                'f1_score': latest.f1_score,
                'confidence_avg': latest.confidence_avg
            }
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'confidence_avg': 0.0}
    
    def get_latest_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get latest performance object for a model"""
        if model_name in self.performance_history and self.performance_history[model_name]:
            return self.performance_history[model_name][-1]
        return None
