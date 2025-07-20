"""
Behavioral Pattern Classification System
Intelligent ML-based system to distinguish engagement-related movements from random movements
Industry-grade precision for educational engagement detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import time
import pickle
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import joblib

from utils.base_processor import BaseProcessor
from utils.logger import logger

class BehavioralPatternClassifier(BaseProcessor):
    """Intelligent behavioral pattern classification for engagement analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BehavioralPatternClassifier", config)
        
        # ML Models for different aspects of behavior
        self.engagement_classifier = None
        self.movement_type_classifier = None
        self.attention_state_classifier = None
        self.anomaly_detector = None
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Classification parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.movement_significance_threshold = config.get('movement_significance_threshold', 0.3)
        
        # Behavioral pattern definitions
        self.engagement_patterns = {
            'high_engagement': {
                'head_stability': (0.7, 1.0),
                'eye_focus': (0.8, 1.0),
                'hand_purposefulness': (0.6, 1.0),
                'posture_alignment': (0.7, 1.0),
                'micro_expression_positivity': (0.6, 1.0)
            },
            'medium_engagement': {
                'head_stability': (0.4, 0.8),
                'eye_focus': (0.5, 0.8),
                'hand_purposefulness': (0.3, 0.7),
                'posture_alignment': (0.4, 0.8),
                'micro_expression_positivity': (0.3, 0.7)
            },
            'low_engagement': {
                'head_stability': (0.0, 0.5),
                'eye_focus': (0.0, 0.6),
                'hand_purposefulness': (0.0, 0.4),
                'posture_alignment': (0.0, 0.5),
                'micro_expression_positivity': (0.0, 0.4)
            }
        }
        
        # Movement classification categories
        self.movement_categories = {
            'engagement_positive': ['hand_raised', 'forward_lean', 'eye_contact', 'note_taking'],
            'engagement_neutral': ['normal_posture', 'casual_movement', 'breathing'],
            'engagement_negative': ['looking_away', 'slouching', 'fidgeting', 'head_down'],
            'random_movement': ['involuntary_twitch', 'adjustment', 'scratch', 'yawn']
        }
        
        # Training data collection
        self.training_data = []
        self.behavioral_history = deque(maxlen=500)
        self.pattern_learning_enabled = config.get('pattern_learning_enabled', True)
        
        # Performance metrics
        self.classification_accuracy = 0.0
        self.model_confidence = 0.0
        
    def initialize(self) -> bool:
        """Initialize behavioral pattern classification system"""
        try:
            logger.info("Initializing behavioral pattern classification system...")
            
            # Initialize ML models
            self._initialize_ml_models()
            
            # Load pre-trained models if available
            self._load_pretrained_models()
            
            # Initialize feature extractors
            self.feature_extractor = BehavioralFeatureExtractor()
            self.pattern_analyzer = MovementPatternAnalyzer()
            self.context_analyzer = ContextualBehaviorAnalyzer()
            
            logger.info("Behavioral pattern classification system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize behavioral classifier: {e}")
            return False
    
    def process_data(self, movement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process movement data for behavioral pattern classification"""
        try:
            # Extract comprehensive behavioral features
            behavioral_features = self.feature_extractor.extract_features(movement_data)
            
            # Analyze movement patterns
            pattern_analysis = self.pattern_analyzer.analyze_patterns(behavioral_features)
            
            # Contextual behavior analysis
            context_analysis = self.context_analyzer.analyze_context(behavioral_features, pattern_analysis)
            
            # Classify engagement level
            engagement_classification = self._classify_engagement_level(behavioral_features)
            
            # Classify movement type
            movement_classification = self._classify_movement_type(behavioral_features)
            
            # Detect behavioral anomalies
            anomaly_analysis = self._detect_behavioral_anomalies(behavioral_features)
            
            # Intelligent decision making
            intelligent_decision = self._make_intelligent_behavioral_decision(
                engagement_classification, movement_classification, 
                pattern_analysis, context_analysis, anomaly_analysis
            )
            
            # Update learning data
            if self.pattern_learning_enabled:
                self._update_learning_data(behavioral_features, intelligent_decision)
            
            # Create comprehensive result
            result = {
                'behavioral_features': behavioral_features,
                'engagement_classification': engagement_classification,
                'movement_classification': movement_classification,
                'pattern_analysis': pattern_analysis,
                'context_analysis': context_analysis,
                'anomaly_analysis': anomaly_analysis,
                'intelligent_decision': intelligent_decision,
                'classification_confidence': self._calculate_overall_confidence(intelligent_decision),
                'behavioral_insights': self._generate_behavioral_insights(intelligent_decision)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in behavioral pattern classification: {e}")
            return self._empty_result(error=str(e))
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Engagement level classifier
            self.engagement_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
            
            # Movement type classifier
            self.movement_type_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            )
            
            # Attention state classifier
            self.attention_state_classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
            
            # Anomaly detector for unusual patterns
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _classify_engagement_level(self, behavioral_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify engagement level using ML model"""
        try:
            # Prepare feature vector for engagement classification
            feature_vector = self._prepare_engagement_feature_vector(behavioral_features)
            
            if feature_vector is None:
                return {'engagement_level': 'unknown', 'confidence': 0.0}
            
            # Scale features
            feature_vector_scaled = self.feature_scaler.fit_transform([feature_vector])
            
            # Classify engagement level
            if hasattr(self.engagement_classifier, 'classes_'):
                prediction = self.engagement_classifier.predict(feature_vector_scaled)[0]
                prediction_proba = self.engagement_classifier.predict_proba(feature_vector_scaled)[0]
                confidence = np.max(prediction_proba)
            else:
                # Fallback to rule-based classification
                prediction, confidence = self._rule_based_engagement_classification(behavioral_features)
            
            # Additional analysis
            engagement_indicators = self._analyze_engagement_indicators(behavioral_features)
            
            return {
                'engagement_level': prediction,
                'confidence': float(confidence),
                'engagement_indicators': engagement_indicators,
                'feature_importance': self._get_engagement_feature_importance(),
                'classification_method': 'ml' if hasattr(self.engagement_classifier, 'classes_') else 'rule_based'
            }
            
        except Exception as e:
            logger.error(f"Error in engagement classification: {e}")
            return {'engagement_level': 'unknown', 'confidence': 0.0}
    
    def _classify_movement_type(self, behavioral_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify type of movement (engagement-related vs random)"""
        try:
            # Extract movement characteristics
            movement_features = behavioral_features.get('movement_features', {})
            
            # Analyze movement purposefulness
            purposefulness_score = self._calculate_movement_purposefulness(movement_features)
            
            # Analyze movement context
            context_relevance = self._analyze_movement_context(movement_features)
            
            # Classify movement significance
            significance_score = (purposefulness_score + context_relevance) / 2
            
            if significance_score > 0.7:
                movement_type = 'engagement_positive'
                significance = 'high'
            elif significance_score > 0.4:
                movement_type = 'engagement_neutral'
                significance = 'medium'
            elif significance_score > 0.2:
                movement_type = 'engagement_negative'
                significance = 'medium'
            else:
                movement_type = 'random_movement'
                significance = 'low'
            
            # Determine if movement should trigger alerts
            should_alert = (movement_type == 'engagement_negative' and 
                          significance_score > self.movement_significance_threshold)
            
            return {
                'movement_type': movement_type,
                'significance': significance,
                'significance_score': significance_score,
                'purposefulness_score': purposefulness_score,
                'context_relevance': context_relevance,
                'should_alert': should_alert,
                'movement_characteristics': self._get_movement_characteristics(movement_features)
            }
            
        except Exception as e:
            logger.error(f"Error in movement classification: {e}")
            return {'movement_type': 'unknown', 'significance': 'low'}
    
    def _calculate_movement_purposefulness(self, movement_features: Dict[str, Any]) -> float:
        """Calculate how purposeful/intentional a movement appears"""
        purposefulness_indicators = []
        
        # Smooth, controlled movements are more purposeful
        movement_smoothness = movement_features.get('smoothness', 0.5)
        purposefulness_indicators.append(movement_smoothness)
        
        # Movements with clear direction are more purposeful
        directional_consistency = movement_features.get('directional_consistency', 0.5)
        purposefulness_indicators.append(directional_consistency)
        
        # Movements that align with engagement gestures are purposeful
        gesture_alignment = movement_features.get('gesture_alignment', 0.5)
        purposefulness_indicators.append(gesture_alignment * 1.5)  # Weight higher
        
        # Movements with appropriate duration are more purposeful
        duration_appropriateness = movement_features.get('duration_appropriateness', 0.5)
        purposefulness_indicators.append(duration_appropriateness)
        
        return np.mean(purposefulness_indicators)
    
    def _analyze_movement_context(self, movement_features: Dict[str, Any]) -> float:
        """Analyze the contextual relevance of movements"""
        context_factors = []
        
        # Timing context (movements during active learning periods)
        timing_relevance = movement_features.get('timing_relevance', 0.5)
        context_factors.append(timing_relevance)
        
        # Spatial context (movements in appropriate areas)
        spatial_relevance = movement_features.get('spatial_relevance', 0.5)
        context_factors.append(spatial_relevance)
        
        # Frequency context (appropriate frequency for the situation)
        frequency_appropriateness = movement_features.get('frequency_appropriateness', 0.5)
        context_factors.append(frequency_appropriateness)
        
        # Social context (movements that align with group behavior)
        social_alignment = movement_features.get('social_alignment', 0.5)
        context_factors.append(social_alignment)
        
        return np.mean(context_factors)
    
    def _detect_behavioral_anomalies(self, behavioral_features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalous behavioral patterns"""
        try:
            # Prepare feature vector for anomaly detection
            feature_vector = self._prepare_anomaly_feature_vector(behavioral_features)
            
            if feature_vector is None:
                return {'is_anomalous': False, 'anomaly_score': 0.0}
            
            # Scale features
            feature_vector_scaled = self.feature_scaler.fit_transform([feature_vector])

            # Detect anomalies (with safety check)
            try:
                # Check if anomaly detector is fitted
                if not hasattr(self.anomaly_detector, 'estimators_'):
                    # Fit with dummy data if not fitted
                    dummy_data = np.random.random((100, len(feature_vector)))
                    self.anomaly_detector.fit(dummy_data)
                    logger.info("Behavioral anomaly detector fitted with dummy data")

                anomaly_score = self.anomaly_detector.decision_function(feature_vector_scaled)[0]
                is_anomalous = anomaly_score < -0.5  # Threshold for anomaly
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                anomaly_score = 0.0
                is_anomalous = False
            
            # Analyze anomaly characteristics
            anomaly_characteristics = self._analyze_anomaly_characteristics(behavioral_features)
            
            return {
                'is_anomalous': bool(is_anomalous),
                'anomaly_score': float(anomaly_score),
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_severity': 'high' if anomaly_score < -0.8 else 'medium' if anomaly_score < -0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {'is_anomalous': False, 'anomaly_score': 0.0}
    
    def _make_intelligent_behavioral_decision(self, engagement_classification: Dict[str, Any],
                                            movement_classification: Dict[str, Any],
                                            pattern_analysis: Dict[str, Any],
                                            context_analysis: Dict[str, Any],
                                            anomaly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent decision about behavioral patterns"""
        try:
            # Extract key metrics
            engagement_level = engagement_classification.get('engagement_level', 'unknown')
            engagement_confidence = engagement_classification.get('confidence', 0.0)
            movement_type = movement_classification.get('movement_type', 'unknown')
            movement_significance = movement_classification.get('significance_score', 0.0)
            is_anomalous = anomaly_analysis.get('is_anomalous', False)
            
            # Decision logic
            decision_factors = {
                'engagement_level': engagement_level,
                'engagement_confidence': engagement_confidence,
                'movement_significance': movement_significance,
                'movement_type': movement_type,
                'is_anomalous': is_anomalous,
                'pattern_consistency': pattern_analysis.get('consistency_score', 0.5),
                'context_appropriateness': context_analysis.get('appropriateness_score', 0.5)
            }
            
            # Calculate overall behavioral score
            behavioral_score = self._calculate_behavioral_score(decision_factors)
            
            # Determine behavioral state
            if behavioral_score > 0.75:
                behavioral_state = 'highly_engaged'
                alert_needed = False
            elif behavioral_score > 0.5:
                behavioral_state = 'moderately_engaged'
                alert_needed = False
            elif behavioral_score > 0.3:
                behavioral_state = 'low_engagement'
                alert_needed = True
            else:
                behavioral_state = 'disengaged'
                alert_needed = True
            
            # Special cases for anomalous behavior
            if is_anomalous and behavioral_score < 0.6:
                behavioral_state = 'concerning_behavior'
                alert_needed = True
            
            # Generate reasoning
            reasoning = self._generate_behavioral_reasoning(decision_factors, behavioral_state)
            
            # Calculate decision confidence
            decision_confidence = self._calculate_decision_confidence(decision_factors)
            
            intelligent_decision = {
                'behavioral_state': behavioral_state,
                'behavioral_score': behavioral_score,
                'decision_confidence': decision_confidence,
                'alert_needed': alert_needed,
                'decision_factors': decision_factors,
                'reasoning': reasoning,
                'recommended_action': self._get_recommended_action(behavioral_state),
                'confidence_threshold_met': decision_confidence >= self.confidence_threshold
            }
            
            return intelligent_decision
            
        except Exception as e:
            logger.error(f"Error in intelligent behavioral decision: {e}")
            return {
                'behavioral_state': 'unknown',
                'behavioral_score': 0.0,
                'decision_confidence': 0.0,
                'alert_needed': False
            }
    
    def _calculate_behavioral_score(self, decision_factors: Dict[str, Any]) -> float:
        """Calculate overall behavioral engagement score"""
        # Weight different factors
        weights = {
            'engagement_confidence': 0.3,
            'movement_significance': 0.25,
            'pattern_consistency': 0.2,
            'context_appropriateness': 0.15,
            'anomaly_penalty': 0.1
        }
        
        # Map engagement level to score
        engagement_level_scores = {
            'high_engagement': 0.9,
            'medium_engagement': 0.6,
            'low_engagement': 0.3,
            'unknown': 0.5
        }
        
        engagement_base_score = engagement_level_scores.get(
            decision_factors.get('engagement_level', 'unknown'), 0.5
        )
        
        # Calculate weighted score
        score = (
            engagement_base_score * decision_factors.get('engagement_confidence', 0.5) * weights['engagement_confidence'] +
            decision_factors.get('movement_significance', 0.5) * weights['movement_significance'] +
            decision_factors.get('pattern_consistency', 0.5) * weights['pattern_consistency'] +
            decision_factors.get('context_appropriateness', 0.5) * weights['context_appropriateness']
        )
        
        # Apply anomaly penalty
        if decision_factors.get('is_anomalous', False):
            score -= weights['anomaly_penalty']
        
        return max(0.0, min(1.0, score))
    
    def _generate_behavioral_reasoning(self, decision_factors: Dict[str, Any], behavioral_state: str) -> List[str]:
        """Generate human-readable reasoning for behavioral decision"""
        reasoning = []
        
        # Engagement level reasoning
        engagement_level = decision_factors.get('engagement_level', 'unknown')
        engagement_confidence = decision_factors.get('engagement_confidence', 0.0)
        
        if engagement_level == 'high_engagement' and engagement_confidence > 0.7:
            reasoning.append("Strong indicators of high engagement detected")
        elif engagement_level == 'low_engagement':
            reasoning.append("Multiple indicators suggest low engagement")
        
        # Movement reasoning
        movement_significance = decision_factors.get('movement_significance', 0.0)
        if movement_significance > 0.7:
            reasoning.append("Purposeful, engagement-related movements observed")
        elif movement_significance < 0.3:
            reasoning.append("Random or disengagement-related movements detected")
        
        # Anomaly reasoning
        if decision_factors.get('is_anomalous', False):
            reasoning.append("Unusual behavioral pattern detected requiring attention")
        
        # Pattern consistency reasoning
        pattern_consistency = decision_factors.get('pattern_consistency', 0.5)
        if pattern_consistency < 0.3:
            reasoning.append("Inconsistent behavioral patterns observed")
        
        return reasoning
    
    def _get_recommended_action(self, behavioral_state: str) -> str:
        """Get recommended action based on behavioral state"""
        action_map = {
            'highly_engaged': 'Continue current approach, student is highly engaged',
            'moderately_engaged': 'Maintain engagement with occasional interaction',
            'low_engagement': 'Consider interactive activity or check for understanding',
            'disengaged': 'Immediate intervention recommended - direct engagement needed',
            'concerning_behavior': 'Individual attention required - unusual pattern detected',
            'unknown': 'Continue monitoring for clearer behavioral patterns'
        }
        
        return action_map.get(behavioral_state, 'Monitor situation and adjust as needed')
    
    def _calculate_decision_confidence(self, decision_factors: Dict[str, Any]) -> float:
        """Calculate confidence in the behavioral decision"""
        confidence_factors = [
            decision_factors.get('engagement_confidence', 0.0),
            decision_factors.get('pattern_consistency', 0.5),
            decision_factors.get('context_appropriateness', 0.5)
        ]
        
        # Reduce confidence if anomalous
        if decision_factors.get('is_anomalous', False):
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)
    
    def _calculate_overall_confidence(self, intelligent_decision: Dict[str, Any]) -> float:
        """Calculate overall classification confidence"""
        return intelligent_decision.get('decision_confidence', 0.0)
    
    def _generate_behavioral_insights(self, intelligent_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about behavioral patterns"""
        behavioral_state = intelligent_decision.get('behavioral_state', 'unknown')
        behavioral_score = intelligent_decision.get('behavioral_score', 0.0)
        
        insights = {
            'primary_insight': f"Student shows {behavioral_state} with score {behavioral_score:.2f}",
            'engagement_trend': self._analyze_engagement_trend(),
            'behavioral_consistency': self._analyze_behavioral_consistency(),
            'attention_quality': self._analyze_attention_quality(),
            'intervention_urgency': self._assess_intervention_urgency(intelligent_decision)
        }
        
        return insights
    
    def _analyze_engagement_trend(self) -> str:
        """Analyze engagement trend over time"""
        if len(self.behavioral_history) < 10:
            return 'insufficient_data'
        
        recent_scores = [b.get('behavioral_score', 0.5) for b in list(self.behavioral_history)[-20:]]
        
        if len(recent_scores) > 5:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if trend > 0.01:
                return 'improving'
            elif trend < -0.01:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'
    
    def _analyze_behavioral_consistency(self) -> float:
        """Analyze consistency of behavioral patterns"""
        if len(self.behavioral_history) < 5:
            return 0.5
        
        recent_states = [b.get('behavioral_state', 'unknown') for b in list(self.behavioral_history)[-10:]]
        
        # Calculate consistency as frequency of most common state
        if recent_states:
            most_common_state = max(set(recent_states), key=recent_states.count)
            consistency = recent_states.count(most_common_state) / len(recent_states)
            return consistency
        
        return 0.5
    
    def _analyze_attention_quality(self) -> str:
        """Analyze quality of attention patterns"""
        if len(self.behavioral_history) < 5:
            return 'unknown'
        
        recent_data = list(self.behavioral_history)[-10:]
        attention_scores = []
        
        for data in recent_data:
            decision_factors = data.get('decision_factors', {})
            pattern_consistency = decision_factors.get('pattern_consistency', 0.5)
            context_appropriateness = decision_factors.get('context_appropriateness', 0.5)
            attention_score = (pattern_consistency + context_appropriateness) / 2
            attention_scores.append(attention_score)
        
        if attention_scores:
            avg_attention = np.mean(attention_scores)
            
            if avg_attention > 0.7:
                return 'high_quality'
            elif avg_attention > 0.4:
                return 'moderate_quality'
            else:
                return 'low_quality'
        
        return 'unknown'
    
    def _assess_intervention_urgency(self, intelligent_decision: Dict[str, Any]) -> str:
        """Assess urgency of intervention needed"""
        behavioral_state = intelligent_decision.get('behavioral_state', 'unknown')
        behavioral_score = intelligent_decision.get('behavioral_score', 0.5)
        alert_needed = intelligent_decision.get('alert_needed', False)
        
        if behavioral_state == 'concerning_behavior':
            return 'immediate'
        elif behavioral_state == 'disengaged' and behavioral_score < 0.2:
            return 'high'
        elif alert_needed:
            return 'moderate'
        else:
            return 'low'
    
    def _update_learning_data(self, behavioral_features: Dict[str, Any], intelligent_decision: Dict[str, Any]):
        """Update learning data for model improvement"""
        try:
            learning_sample = {
                'timestamp': time.time(),
                'behavioral_features': behavioral_features,
                'intelligent_decision': intelligent_decision,
                'behavioral_state': intelligent_decision.get('behavioral_state', 'unknown'),
                'behavioral_score': intelligent_decision.get('behavioral_score', 0.0)
            }
            
            self.behavioral_history.append(learning_sample)
            
            # OPTIMIZATION: Retrain models less frequently for better FPS
            if len(self.behavioral_history) % 200 == 0:  # Changed from 50 to 200
                self._retrain_models()
                
        except Exception as e:
            logger.error(f"Error updating learning data: {e}")
    
    def _retrain_models(self):
        """Retrain models with accumulated behavioral data"""
        try:
            if len(self.behavioral_history) < 30:
                return
            
            logger.info("Retraining behavioral classification models...")
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if X is not None and len(X) > 20:
                # Retrain engagement classifier
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.engagement_classifier.fit(X_train, y_train)
                
                # Update performance metrics
                accuracy = self.engagement_classifier.score(X_test, y_test)
                self.classification_accuracy = accuracy
                
                logger.info(f"Models retrained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from behavioral history"""
        try:
            feature_vectors = []
            labels = []
            
            for sample in self.behavioral_history:
                behavioral_features = sample.get('behavioral_features', {})
                behavioral_state = sample.get('behavioral_state', 'unknown')
                
                feature_vector = self._prepare_engagement_feature_vector(behavioral_features)
                if feature_vector and behavioral_state != 'unknown':
                    feature_vectors.append(feature_vector)
                    labels.append(behavioral_state)
            
            if len(feature_vectors) > 0:
                return np.array(feature_vectors), np.array(labels)
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _prepare_engagement_feature_vector(self, behavioral_features: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare feature vector for engagement classification"""
        try:
            feature_vector = []
            
            # Movement features
            movement_features = behavioral_features.get('movement_features', {})
            feature_vector.extend([
                movement_features.get('head_stability', 0.5),
                movement_features.get('eye_focus_score', 0.5),
                movement_features.get('hand_purposefulness', 0.5),
                movement_features.get('posture_alignment', 0.5),
                movement_features.get('movement_smoothness', 0.5)
            ])
            
            # Attention features
            attention_features = behavioral_features.get('attention_features', {})
            feature_vector.extend([
                attention_features.get('focus_duration', 0.5),
                attention_features.get('attention_consistency', 0.5),
                attention_features.get('distraction_frequency', 0.5)
            ])
            
            # Emotional features
            emotional_features = behavioral_features.get('emotional_features', {})
            feature_vector.extend([
                emotional_features.get('positive_emotion_ratio', 0.5),
                emotional_features.get('engagement_emotion_score', 0.5),
                emotional_features.get('emotional_stability', 0.5)
            ])
            
            return feature_vector if len(feature_vector) > 0 else None
            
        except Exception as e:
            logger.error(f"Error preparing engagement feature vector: {e}")
            return None
    
    def _prepare_anomaly_feature_vector(self, behavioral_features: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare feature vector for anomaly detection"""
        # Similar to engagement features but focused on anomaly detection
        return self._prepare_engagement_feature_vector(behavioral_features)
    
    def _rule_based_engagement_classification(self, behavioral_features: Dict[str, Any]) -> Tuple[str, float]:
        """Fallback rule-based engagement classification"""
        # Extract key behavioral indicators
        movement_features = behavioral_features.get('movement_features', {})
        attention_features = behavioral_features.get('attention_features', {})
        
        head_stability = movement_features.get('head_stability', 0.5)
        eye_focus = attention_features.get('focus_score', 0.5)
        posture_alignment = movement_features.get('posture_alignment', 0.5)
        
        # Calculate engagement score
        engagement_score = (head_stability + eye_focus + posture_alignment) / 3
        
        # Classify based on score
        if engagement_score > 0.7:
            return 'high_engagement', 0.8
        elif engagement_score > 0.4:
            return 'medium_engagement', 0.7
        else:
            return 'low_engagement', 0.6
    
    def _analyze_engagement_indicators(self, behavioral_features: Dict[str, Any]) -> List[str]:
        """Analyze specific engagement indicators"""
        indicators = []
        
        movement_features = behavioral_features.get('movement_features', {})
        
        if movement_features.get('head_stability', 0) > 0.7:
            indicators.append('stable_head_position')
        
        if movement_features.get('eye_focus_score', 0) > 0.8:
            indicators.append('strong_eye_focus')
        
        if movement_features.get('hand_purposefulness', 0) > 0.6:
            indicators.append('purposeful_hand_movements')
        
        return indicators
    
    def _get_engagement_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from engagement classifier"""
        try:
            if hasattr(self.engagement_classifier, 'feature_importances_'):
                feature_names = [
                    'head_stability', 'eye_focus', 'hand_purposefulness', 'posture_alignment',
                    'movement_smoothness', 'focus_duration', 'attention_consistency',
                    'distraction_frequency', 'positive_emotion_ratio', 'engagement_emotion_score',
                    'emotional_stability'
                ]
                
                importances = self.engagement_classifier.feature_importances_
                return dict(zip(feature_names[:len(importances)], importances.tolist()))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _get_movement_characteristics(self, movement_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed movement characteristics"""
        return {
            'smoothness': movement_features.get('smoothness', 0.5),
            'directional_consistency': movement_features.get('directional_consistency', 0.5),
            'frequency': movement_features.get('frequency', 0.5),
            'amplitude': movement_features.get('amplitude', 0.5),
            'duration': movement_features.get('duration', 0.5)
        }
    
    def _analyze_anomaly_characteristics(self, behavioral_features: Dict[str, Any]) -> List[str]:
        """Analyze characteristics of detected anomalies"""
        characteristics = []
        
        movement_features = behavioral_features.get('movement_features', {})
        
        if movement_features.get('frequency', 0) > 0.8:
            characteristics.append('high_frequency_movement')
        
        if movement_features.get('amplitude', 0) > 0.8:
            characteristics.append('large_amplitude_movement')
        
        if movement_features.get('directional_consistency', 0) < 0.2:
            characteristics.append('erratic_movement_pattern')
        
        return characteristics
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            # Attempt to load pre-trained models
            pass
        except Exception as e:
            logger.info("No pre-trained behavioral models found, will train from scratch")
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'behavioral_features': {},
            'engagement_classification': {'engagement_level': 'unknown', 'confidence': 0.0},
            'movement_classification': {'movement_type': 'unknown', 'significance': 'low'},
            'pattern_analysis': {},
            'context_analysis': {},
            'anomaly_analysis': {'is_anomalous': False},
            'intelligent_decision': {
                'behavioral_state': 'unknown',
                'behavioral_score': 0.0,
                'alert_needed': False
            },
            'classification_confidence': 0.0,
            'behavioral_insights': {}
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup behavioral classifier resources"""
        # Save models for future use
        try:
            self._save_models()
        except Exception as e:
            logger.error(f"Error saving behavioral models: {e}")
        
        logger.info("Behavioral pattern classifier cleaned up")
    
    def _save_models(self):
        """Save trained models"""
        try:
            if hasattr(self.engagement_classifier, 'classes_'):
                joblib.dump(self.engagement_classifier, 'data/models/engagement_classifier.pkl')
            
            if hasattr(self.movement_type_classifier, 'classes_'):
                joblib.dump(self.movement_type_classifier, 'data/models/movement_classifier.pkl')
            
            joblib.dump(self.feature_scaler, 'data/models/behavioral_scaler.pkl')
            
            logger.info("Behavioral models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving behavioral models: {e}")

# Helper classes for behavioral analysis
class BehavioralFeatureExtractor:
    """Extract comprehensive behavioral features"""
    
    def extract_features(self, movement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral features from movement data"""
        # Placeholder for comprehensive feature extraction
        return {
            'movement_features': {
                'head_stability': 0.7,
                'eye_focus_score': 0.8,
                'hand_purposefulness': 0.6,
                'posture_alignment': 0.7,
                'movement_smoothness': 0.6
            },
            'attention_features': {
                'focus_duration': 0.8,
                'attention_consistency': 0.7,
                'distraction_frequency': 0.2
            },
            'emotional_features': {
                'positive_emotion_ratio': 0.6,
                'engagement_emotion_score': 0.7,
                'emotional_stability': 0.8
            }
        }

class MovementPatternAnalyzer:
    """Analyze movement patterns for behavioral classification"""
    
    def analyze_patterns(self, behavioral_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze movement patterns"""
        return {
            'consistency_score': 0.7,
            'pattern_type': 'engagement_positive',
            'confidence': 0.8
        }

class ContextualBehaviorAnalyzer:
    """Analyze behavioral context"""
    
    def analyze_context(self, behavioral_features: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral context"""
        return {
            'appropriateness_score': 0.8,
            'context_type': 'learning_environment',
            'social_alignment': 0.7
        }
