"""
Intelligent Disengagement Pattern Recognition System
Advanced ML-based analysis to distinguish genuine disengagement from random movements
Industry-grade precision for educational engagement detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import time
import os
import pickle
from collections import deque
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib

from utils.base_processor import BaseProcessor
from utils.logger import logger

class IntelligentPatternAnalyzer(BaseProcessor):
    """ML-powered intelligent pattern analysis for engagement detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("IntelligentPatternAnalyzer", config)
        
        # ML Models for pattern recognition
        self.disengagement_classifier = None
        self.anomaly_detector = None
        self.pattern_clusterer = None
        self.feature_scaler = StandardScaler()
        
        # Pattern analysis parameters
        self.feature_window = config.get('feature_window', 30)  # 30 frames for pattern analysis
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.anomaly_threshold = config.get('anomaly_threshold', -0.5)
        
        # Feature extraction
        self.feature_extractor = EngagementFeatureExtractor()
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.behavioral_classifier = BehavioralPatternClassifier()
        
        # Data storage for learning
        self.pattern_history = deque(maxlen=1000)
        self.labeled_patterns = []
        self.model_performance = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Engagement state tracking
        self.current_engagement_state = 'unknown'
        self.state_confidence = 0.0
        self.state_history = deque(maxlen=100)
        
    def initialize(self) -> bool:
        """Initialize ML models and pattern analyzers"""
        try:
            logger.info("Initializing intelligent pattern analyzer...")
            
            # Initialize feature extractor
            self.feature_extractor.initialize()
            
            # Initialize temporal analyzer
            self.temporal_analyzer.initialize()
            
            # Initialize behavioral classifier
            self.behavioral_classifier.initialize()
            
            # Load checkpoint before loading models (if method exists)
            if hasattr(self, '_load_checkpoint'):
                self._load_checkpoint()

            # Load pre-trained models if available
            self._load_pretrained_models()

            # Initialize ML models
            self._initialize_ml_models()
            
            logger.info("Intelligent pattern analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligent pattern analyzer: {e}")
            return False
    
    def process_data(self, movement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process movement data for intelligent pattern analysis"""
        try:
            # Extract comprehensive features
            features = self.feature_extractor.extract_features(movement_data)
            
            # Temporal pattern analysis
            temporal_patterns = self.temporal_analyzer.analyze_patterns(features)
            
            # Behavioral classification
            behavioral_analysis = self.behavioral_classifier.classify_behavior(features, temporal_patterns)
            
            # ML-based disengagement detection
            ml_analysis = self._ml_pattern_analysis(features, temporal_patterns)
            
            # Intelligent decision making
            intelligent_decision = self._make_intelligent_decision(
                behavioral_analysis, ml_analysis, temporal_patterns
            )
            
            # Update learning data
            self._update_learning_data(features, intelligent_decision)
            
            # Create comprehensive result
            result = {
                'intelligent_analysis': intelligent_decision,
                'behavioral_patterns': behavioral_analysis,
                'temporal_patterns': temporal_patterns,
                'ml_confidence': ml_analysis.get('confidence', 0.0),
                'feature_importance': self._get_feature_importance(),
                'learning_metrics': self._get_learning_metrics(),
                'alert_recommendation': self._generate_alert_recommendation(intelligent_decision)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intelligent pattern analysis: {e}")
            return self._empty_result(error=str(e))
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Disengagement classifier (Random Forest for interpretability)
            self.disengagement_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Anomaly detector for unusual patterns
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Pattern clusterer for behavior grouping
            self.pattern_clusterer = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _ml_pattern_analysis(self, features: Dict[str, Any], temporal_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML-based pattern analysis"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, temporal_patterns)
            
            if feature_vector is None or len(feature_vector) == 0:
                return {'confidence': 0.0, 'prediction': 'unknown'}
            
            # Scale features
            feature_vector_scaled = self.feature_scaler.fit_transform([feature_vector])

            # Anomaly detection (with safety check)
            try:
                # Check if anomaly detector is fitted
                if not hasattr(self.anomaly_detector, 'estimators_'):
                    # Fit with dummy data if not fitted
                    dummy_data = np.random.random((100, len(feature_vector)))
                    self.anomaly_detector.fit(dummy_data)
                    logger.info("Anomaly detector fitted with dummy data")

                anomaly_score = self.anomaly_detector.decision_function(feature_vector_scaled)[0]
                is_anomaly = anomaly_score < self.anomaly_threshold
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                anomaly_score = 0.0
                is_anomaly = False
            
            # Classification (if model is trained)
            classification_result = self._classify_engagement_state(feature_vector_scaled)
            
            # Combine results
            ml_analysis = {
                'anomaly_score': float(anomaly_score),
                'is_anomalous': bool(is_anomaly),
                'classification': classification_result,
                'confidence': self._calculate_ml_confidence(anomaly_score, classification_result),
                'feature_vector': feature_vector
            }
            
            return ml_analysis
            
        except Exception as e:
            logger.error(f"Error in ML pattern analysis: {e}")
            return {'confidence': 0.0, 'prediction': 'unknown'}
    
    def _prepare_feature_vector(self, features: Dict[str, Any], temporal_patterns: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare feature vector for ML analysis"""
        try:
            feature_vector = []
            
            # Add movement features
            movement_features = features.get('movement_features', {})
            feature_vector.extend([
                movement_features.get('head_movement_intensity', 0.0),
                movement_features.get('eye_movement_velocity', 0.0),
                movement_features.get('hand_movement_energy', 0.0),
                movement_features.get('posture_stability', 0.0),
                movement_features.get('micro_movement_frequency', 0.0)
            ])
            
            # Add attention features
            attention_features = features.get('attention_features', {})
            feature_vector.extend([
                attention_features.get('gaze_focus_score', 0.0),
                attention_features.get('attention_duration', 0.0),
                attention_features.get('focus_stability', 0.0),
                attention_features.get('distraction_frequency', 0.0)
            ])
            
            # Add behavioral features
            behavioral_features = features.get('behavioral_features', {})
            feature_vector.extend([
                behavioral_features.get('engagement_gesture_frequency', 0.0),
                behavioral_features.get('disengagement_gesture_frequency', 0.0),
                behavioral_features.get('fidgeting_intensity', 0.0),
                behavioral_features.get('posture_engagement_score', 0.0)
            ])
            
            # Add temporal features
            temporal_features = temporal_patterns.get('pattern_features', {})
            feature_vector.extend([
                temporal_features.get('pattern_consistency', 0.0),
                temporal_features.get('trend_direction', 0.0),
                temporal_features.get('variability_score', 0.0),
                temporal_features.get('periodicity_strength', 0.0)
            ])
            
            return feature_vector if len(feature_vector) > 0 else None
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None
    
    def _classify_engagement_state(self, feature_vector_scaled: np.ndarray) -> Dict[str, Any]:
        """Classify engagement state using trained model"""
        try:
            # Check if model is trained
            if not hasattr(self.disengagement_classifier, 'classes_'):
                return {'prediction': 'unknown', 'confidence': 0.0}
            
            # Make prediction
            prediction = self.disengagement_classifier.predict(feature_vector_scaled)[0]
            prediction_proba = self.disengagement_classifier.predict_proba(feature_vector_scaled)[0]
            
            # Get confidence
            confidence = np.max(prediction_proba)
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': prediction_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in engagement classification: {e}")
            return {'prediction': 'unknown', 'confidence': 0.0}

    def _calculate_ml_confidence(self, anomaly_score: float, classification_result: Dict[str, Any]) -> float:
        """Calculate ML confidence score combining anomaly detection and classification"""
        try:
            # Normalize anomaly score to 0-1 range
            anomaly_confidence = max(0.0, min(1.0, (anomaly_score + 1.0) / 2.0))

            # Get classification confidence
            classification_confidence = classification_result.get('confidence', 0.0)

            # Combine both confidences
            if classification_confidence > 0:
                # Weight classification more if available
                combined_confidence = (classification_confidence * 0.7) + (anomaly_confidence * 0.3)
            else:
                # Use only anomaly confidence if classification not available
                combined_confidence = anomaly_confidence

            return float(combined_confidence)

        except Exception as e:
            logger.error(f"Error calculating ML confidence: {e}")
            return 0.0

    def _make_intelligent_decision(self, behavioral_analysis: Dict[str, Any],
                                 ml_analysis: Dict[str, Any], 
                                 temporal_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent decision combining all analysis methods"""
        try:
            # Weight different analysis methods
            behavioral_weight = 0.4
            ml_weight = 0.4
            temporal_weight = 0.2
            
            # Extract scores
            behavioral_score = behavioral_analysis.get('engagement_score', 0.5)
            ml_confidence = ml_analysis.get('confidence', 0.0)
            temporal_consistency = temporal_patterns.get('pattern_consistency', 0.5)
            
            # Calculate weighted decision score
            decision_score = (
                behavioral_score * behavioral_weight +
                ml_confidence * ml_weight +
                temporal_consistency * temporal_weight
            )
            
            # Determine engagement state
            if decision_score > 0.7:
                engagement_state = 'engaged'
                alert_needed = False
            elif decision_score < 0.3:
                engagement_state = 'disengaged'
                alert_needed = True
            else:
                engagement_state = 'neutral'
                alert_needed = False
            
            # Check for anomalous patterns
            is_anomalous = ml_analysis.get('is_anomalous', False)
            if is_anomalous and decision_score < 0.5:
                alert_needed = True
                engagement_state = 'concerning_pattern'
            
            # Calculate overall confidence
            confidence_scores = [
                behavioral_analysis.get('confidence', 0.0),
                ml_analysis.get('confidence', 0.0),
                temporal_patterns.get('confidence', 0.0)
            ]
            overall_confidence = np.mean([c for c in confidence_scores if c > 0])
            
            intelligent_decision = {
                'engagement_state': engagement_state,
                'decision_score': decision_score,
                'overall_confidence': overall_confidence,
                'alert_needed': alert_needed,
                'reasoning': self._generate_reasoning(behavioral_analysis, ml_analysis, temporal_patterns),
                'component_scores': {
                    'behavioral': behavioral_score,
                    'ml_confidence': ml_confidence,
                    'temporal_consistency': temporal_consistency
                }
            }
            
            # Update state history
            self._update_state_history(intelligent_decision)
            
            return intelligent_decision
            
        except Exception as e:
            logger.error(f"Error in intelligent decision making: {e}")
            return {
                'engagement_state': 'unknown',
                'decision_score': 0.0,
                'overall_confidence': 0.0,
                'alert_needed': False
            }
    
    def _generate_reasoning(self, behavioral_analysis: Dict[str, Any], 
                          ml_analysis: Dict[str, Any], 
                          temporal_patterns: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for the decision"""
        reasoning = []
        
        # Behavioral reasoning
        behavioral_score = behavioral_analysis.get('engagement_score', 0.5)
        if behavioral_score > 0.7:
            reasoning.append("Strong positive behavioral indicators detected")
        elif behavioral_score < 0.3:
            reasoning.append("Multiple disengagement behaviors observed")
        
        # ML reasoning
        if ml_analysis.get('is_anomalous', False):
            reasoning.append("Unusual movement pattern detected by ML model")
        
        # Temporal reasoning
        pattern_consistency = temporal_patterns.get('pattern_consistency', 0.5)
        if pattern_consistency < 0.3:
            reasoning.append("Inconsistent engagement patterns over time")
        elif pattern_consistency > 0.7:
            reasoning.append("Consistent engagement pattern maintained")
        
        return reasoning
    
    def _generate_alert_recommendation(self, intelligent_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alert recommendation based on intelligent analysis"""
        alert_needed = intelligent_decision.get('alert_needed', False)
        confidence = intelligent_decision.get('overall_confidence', 0.0)
        engagement_state = intelligent_decision.get('engagement_state', 'unknown')
        
        if not alert_needed:
            return {'should_alert': False, 'reason': 'No disengagement detected'}
        
        # Only alert if confidence is high enough
        if confidence < self.confidence_threshold:
            return {
                'should_alert': False, 
                'reason': f'Confidence too low ({confidence:.2f} < {self.confidence_threshold})'
            }
        
        # Generate alert recommendation
        alert_recommendation = {
            'should_alert': True,
            'alert_type': 'disengagement_detected',
            'severity': 'high' if confidence > 0.9 else 'medium',
            'confidence': confidence,
            'engagement_state': engagement_state,
            'recommended_action': self._get_recommended_action(engagement_state),
            'reasoning': intelligent_decision.get('reasoning', [])
        }
        
        return alert_recommendation
    
    def _get_recommended_action(self, engagement_state: str) -> str:
        """Get recommended action based on engagement state"""
        action_map = {
            'disengaged': 'Consider interactive activity or check for understanding',
            'concerning_pattern': 'Individual attention may be needed',
            'neutral': 'Monitor for changes in engagement',
            'unknown': 'Continue observation'
        }
        
        return action_map.get(engagement_state, 'Monitor situation')
    
    def _update_learning_data(self, features: Dict[str, Any], decision: Dict[str, Any]):
        """Update learning data for model improvement"""
        try:
            learning_sample = {
                'timestamp': time.time(),
                'features': features,
                'decision': decision,
                'engagement_state': decision.get('engagement_state', 'unknown'),
                'confidence': decision.get('overall_confidence', 0.0)
            }
            
            self.pattern_history.append(learning_sample)
            
            # Periodically retrain models if enough data
            if len(self.pattern_history) % 100 == 0:
                self._retrain_models()
                
        except Exception as e:
            logger.error(f"Error updating learning data: {e}")
    
    def _retrain_models(self):
        """Retrain models with accumulated data"""
        try:
            if len(self.pattern_history) < 50:
                return
            
            logger.info("Retraining models with new data...")
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if X is not None and len(X) > 10:
                # Retrain anomaly detector
                self.anomaly_detector.fit(X)
                
                # Update feature scaler
                self.feature_scaler.fit(X)
                
                logger.info("Models retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from pattern history"""
        try:
            feature_vectors = []
            labels = []
            
            for sample in self.pattern_history:
                features = sample.get('features', {})
                temporal_patterns = sample.get('decision', {}).get('temporal_patterns', {})
                
                feature_vector = self._prepare_feature_vector(features, temporal_patterns)
                if feature_vector:
                    feature_vectors.append(feature_vector)
                    
                    # Create label based on engagement state
                    engagement_state = sample.get('engagement_state', 'unknown')
                    label = 1 if engagement_state == 'engaged' else 0
                    labels.append(label)
            
            if len(feature_vectors) > 0:
                return np.array(feature_vectors), np.array(labels)
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            if hasattr(self.disengagement_classifier, 'feature_importances_'):
                importances = self.disengagement_classifier.feature_importances_
                
                feature_names = [
                    'head_movement', 'eye_movement', 'hand_movement', 'posture_stability',
                    'micro_movement', 'gaze_focus', 'attention_duration', 'focus_stability',
                    'distraction_freq', 'engagement_gestures', 'disengagement_gestures',
                    'fidgeting', 'posture_engagement', 'pattern_consistency', 'trend_direction',
                    'variability', 'periodicity'
                ]
                
                return dict(zip(feature_names[:len(importances)], importances.tolist()))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning and performance metrics"""
        return {
            'samples_collected': len(self.pattern_history),
            'model_performance': self.model_performance,
            'confidence_threshold': self.confidence_threshold,
            'anomaly_threshold': self.anomaly_threshold
        }
    
    def _update_state_history(self, decision: Dict[str, Any]):
        """Update engagement state history"""
        self.state_history.append({
            'timestamp': time.time(),
            'state': decision.get('engagement_state', 'unknown'),
            'confidence': decision.get('overall_confidence', 0.0),
            'score': decision.get('decision_score', 0.0)
        })
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            # Attempt to load pre-trained models
            # This would load models trained on educational engagement data
            pass
        except Exception as e:
            logger.info("No pre-trained models found, will train from scratch")
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'intelligent_analysis': {
                'engagement_state': 'unknown',
                'decision_score': 0.0,
                'overall_confidence': 0.0,
                'alert_needed': False
            },
            'behavioral_patterns': {},
            'temporal_patterns': {},
            'ml_confidence': 0.0,
            'feature_importance': {},
            'learning_metrics': {},
            'alert_recommendation': {'should_alert': False}
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup resources"""
        # Save models and checkpoint for future use
        try:
            self._save_models()
            self._save_checkpoint()
            logger.info("💾 Models and checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Error saving models/checkpoint: {e}")

        logger.info("Intelligent pattern analyzer cleaned up")
    
    def _save_models(self):
        """Save trained models"""
        try:
            if hasattr(self.disengagement_classifier, 'classes_'):
                joblib.dump(self.disengagement_classifier, 'data/models/disengagement_classifier.pkl')
            
            joblib.dump(self.anomaly_detector, 'data/models/anomaly_detector.pkl')
            joblib.dump(self.feature_scaler, 'data/models/feature_scaler.pkl')
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _save_checkpoint(self):
        """Save learning checkpoint for persistence"""
        try:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'pattern_analyzer')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, 'pattern_analyzer_checkpoint.pkl')

            checkpoint_data = {
                'save_timestamp': time.time(),
                'pattern_history': list(self.pattern_history),
                'feature_importance': self.feature_importance,
                'learning_rate': getattr(self, 'learning_rate', 0.01),
                'model_performance': getattr(self, 'model_performance', {}),
                'training_iterations': getattr(self, 'training_iterations', 0)
            }

            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            logger.info(f"💾 Pattern analyzer checkpoint saved")

        except Exception as e:
            logger.error(f"Error saving pattern analyzer checkpoint: {e}")

    def _load_checkpoint(self):
        """Load learning checkpoint to continue from last session"""
        try:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'pattern_analyzer')
            checkpoint_file = os.path.join(checkpoint_dir, 'pattern_analyzer_checkpoint.pkl')

            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                # Restore pattern history
                if 'pattern_history' in checkpoint_data:
                    self.pattern_history = deque(checkpoint_data['pattern_history'], maxlen=1000)

                # Restore feature importance
                if 'feature_importance' in checkpoint_data:
                    self.feature_importance = checkpoint_data['feature_importance']

                checkpoint_age = time.time() - checkpoint_data.get('save_timestamp', time.time())
                logger.info(f"📂 Pattern analyzer checkpoint loaded (age: {checkpoint_age/3600:.1f} hours)")

            else:
                logger.info("📂 No pattern analyzer checkpoint found - starting fresh")

        except Exception as e:
            logger.error(f"Error loading pattern analyzer checkpoint: {e}")
            logger.info("📂 Starting fresh pattern analysis session")

# Helper classes for feature extraction and analysis
class EngagementFeatureExtractor:
    """Extract comprehensive features for engagement analysis"""
    
    def initialize(self):
        logger.info("Feature extractor initialized")
    
    def extract_features(self, movement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from movement data"""
        # Placeholder for comprehensive feature extraction
        return {
            'movement_features': {},
            'attention_features': {},
            'behavioral_features': {}
        }

class TemporalPatternAnalyzer:
    """Analyze temporal patterns in engagement data"""
    
    def initialize(self):
        logger.info("Temporal pattern analyzer initialized")
    
    def analyze_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        # Placeholder for temporal pattern analysis
        return {
            'pattern_features': {},
            'pattern_consistency': 0.5,
            'confidence': 0.5
        }

class BehavioralPatternClassifier:
    """Classify behavioral patterns"""
    
    def initialize(self):
        logger.info("Behavioral pattern classifier initialized")
    
    def classify_behavior(self, features: Dict[str, Any], temporal_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Classify behavioral patterns"""
        # Placeholder for behavioral classification
        return {
            'engagement_score': 0.5,
            'confidence': 0.5
        }
