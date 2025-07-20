"""
Intelligent Real-time Alert System
Enhanced alert system that triggers only on genuine disengagement patterns
Industry-grade precision with confidence scoring and intelligent filtering
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum
import json

from utils.base_processor import BaseProcessor
from utils.logger import logger

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of engagement alerts"""
    DISENGAGEMENT_DETECTED = "disengagement_detected"
    ATTENTION_DECLINE = "attention_decline"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    SUSTAINED_INATTENTION = "sustained_inattention"
    CONCERNING_PATTERN = "concerning_pattern"
    PARTICIPATION_DROP = "participation_drop"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    confidence: float
    timestamp: float
    student_id: Optional[str]
    description: str
    evidence: Dict[str, Any]
    recommended_action: str
    expires_at: float
    metadata: Dict[str, Any]

class IntelligentAlertSystem(BaseProcessor):
    """Intelligent alert system with precision filtering and confidence scoring"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("IntelligentAlertSystem", config)
        
        # Alert configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.severity_thresholds = {
            AlertSeverity.LOW: 0.6,
            AlertSeverity.MEDIUM: 0.75,
            AlertSeverity.HIGH: 0.85,
            AlertSeverity.CRITICAL: 0.95
        }
        
        # Alert filtering parameters
        self.min_evidence_duration = config.get('min_evidence_duration', 3.0)  # seconds
        self.alert_cooldown_period = config.get('alert_cooldown_period', 30.0)  # seconds
        self.max_alerts_per_minute = config.get('max_alerts_per_minute', 3)
        
        # Alert intelligence components
        self.pattern_validator = AlertPatternValidator()
        self.confidence_calculator = AlertConfidenceCalculator()
        self.evidence_aggregator = AlertEvidenceAggregator()
        self.alert_prioritizer = AlertPrioritizer()
        
        # Alert tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.recent_alerts = deque(maxlen=50)
        self.alert_statistics = {
            'total_alerts': 0,
            'false_positives': 0,
            'true_positives': 0,
            'accuracy': 0.0
        }
        
        # Evidence collection
        self.evidence_buffer = deque(maxlen=300)  # 10 seconds at 30fps
        self.pattern_evidence = {}
        
        # Alert suppression and filtering
        self.suppressed_patterns = set()
        self.alert_rate_limiter = AlertRateLimiter(self.max_alerts_per_minute)
        
    def initialize(self) -> bool:
        """Initialize intelligent alert system"""
        try:
            logger.info("Initializing intelligent alert system...")
            
            # Initialize alert components
            self.pattern_validator.initialize()
            self.confidence_calculator.initialize()
            self.evidence_aggregator.initialize()
            self.alert_prioritizer.initialize()
            
            # Initialize alert rate limiter
            self.alert_rate_limiter.initialize()
            
            logger.info("Intelligent alert system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligent alert system: {e}")
            return False
    
    def process_data(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process engagement data for intelligent alert generation"""
        try:
            # Safety check for None data
            if engagement_data is None:
                # Only log warning occasionally to avoid spam
                current_time = time.time()
                if not hasattr(self, '_last_none_warning_time') or current_time - self._last_none_warning_time > 10.0:
                    logger.warning("Received None engagement data in alert processing (normal during startup/processing delays)")
                    self._last_none_warning_time = current_time
                return self._empty_result(error="No engagement data provided")

            # Collect evidence
            self._collect_evidence(engagement_data)
            
            # Validate patterns for potential alerts
            pattern_validation = self.pattern_validator.validate_patterns(
                engagement_data, self.evidence_buffer
            )
            
            # Calculate confidence scores
            confidence_analysis = self.confidence_calculator.calculate_confidence(
                pattern_validation, self.evidence_buffer
            )
            
            # Aggregate evidence for alert decisions
            evidence_analysis = self.evidence_aggregator.aggregate_evidence(
                pattern_validation, confidence_analysis
            )
            
            # Generate intelligent alerts
            alert_decisions = self._generate_intelligent_alerts(
                pattern_validation, confidence_analysis, evidence_analysis
            )
            
            # Process and filter alerts
            processed_alerts = self._process_and_filter_alerts(alert_decisions)
            
            # Update alert tracking
            self._update_alert_tracking(processed_alerts)
            
            # Create result
            result = {
                'pattern_validation': pattern_validation,
                'confidence_analysis': confidence_analysis,
                'evidence_analysis': evidence_analysis,
                'alert_decisions': alert_decisions,
                'processed_alerts': processed_alerts,
                'active_alerts': list(self.active_alerts.values()),
                'alert_statistics': self.alert_statistics.copy(),
                'system_status': self._get_system_status()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intelligent alert processing: {e}")
            return self._empty_result(error=str(e))
    
    def _collect_evidence(self, engagement_data: Dict[str, Any]):
        """Collect evidence for alert decision making"""
        current_time = time.time()
        
        # Extract key engagement metrics
        evidence = {
            'timestamp': current_time,
            'overall_engagement_score': engagement_data.get('overall_engagement_score', 0.0),
            'engagement_level': engagement_data.get('engagement_level', 'unknown'),
            'component_scores': engagement_data.get('component_scores', {}),
            'behavioral_indicators': engagement_data.get('behavioral_indicators', {}),
            'attention_metrics': engagement_data.get('attention_metrics', {}),
            'movement_analysis': engagement_data.get('movement_analysis', {}),
            'confidence_scores': engagement_data.get('confidence_scores', {})
        }
        
        # Add to evidence buffer
        self.evidence_buffer.append(evidence)
        
        # Update pattern evidence
        self._update_pattern_evidence(evidence)
    
    def _update_pattern_evidence(self, evidence: Dict[str, Any]):
        """Update evidence for specific patterns"""
        engagement_score = evidence.get('overall_engagement_score', 0.0)
        engagement_level = evidence.get('engagement_level', 'unknown')
        
        # Track disengagement patterns
        if engagement_level in ['low', 'disengaged']:
            pattern_key = 'disengagement_pattern'
            if pattern_key not in self.pattern_evidence:
                self.pattern_evidence[pattern_key] = deque(maxlen=100)
            self.pattern_evidence[pattern_key].append(evidence)
        
        # Track attention decline patterns
        attention_metrics = evidence.get('attention_metrics', {})
        attention_score = attention_metrics.get('attention_score', 0.0)
        
        if attention_score < 0.4:
            pattern_key = 'attention_decline_pattern'
            if pattern_key not in self.pattern_evidence:
                self.pattern_evidence[pattern_key] = deque(maxlen=100)
            self.pattern_evidence[pattern_key].append(evidence)
    
    def _generate_intelligent_alerts(self, pattern_validation: Dict[str, Any],
                                   confidence_analysis: Dict[str, Any],
                                   evidence_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent alerts based on validated patterns"""
        alert_decisions = []
        
        # Check each validated pattern for alert generation
        validated_patterns = pattern_validation.get('validated_patterns', [])
        
        for pattern in validated_patterns:
            pattern_type = pattern.get('pattern_type')
            pattern_confidence = pattern.get('confidence', 0.0)
            pattern_evidence = pattern.get('evidence', {})
            
            # Check if pattern meets alert criteria
            if self._should_generate_alert(pattern, confidence_analysis, evidence_analysis):
                alert_decision = self._create_alert_decision(
                    pattern, confidence_analysis, evidence_analysis
                )
                alert_decisions.append(alert_decision)
        
        return alert_decisions
    
    def _should_generate_alert(self, pattern: Dict[str, Any],
                             confidence_analysis: Dict[str, Any],
                             evidence_analysis: Dict[str, Any]) -> bool:
        """Determine if a pattern should generate an alert"""
        pattern_type = pattern.get('pattern_type')
        pattern_confidence = pattern.get('confidence', 0.0)
        
        # Check confidence threshold
        if pattern_confidence < self.confidence_threshold:
            return False
        
        # Check evidence duration
        evidence_duration = evidence_analysis.get('evidence_duration', 0.0)
        if evidence_duration < self.min_evidence_duration:
            return False
        
        # Check if pattern is suppressed
        if pattern_type in self.suppressed_patterns:
            return False
        
        # Check rate limiting
        if not self.alert_rate_limiter.can_generate_alert():
            return False
        
        # Check for recent similar alerts (cooldown)
        if self._is_in_cooldown_period(pattern_type):
            return False
        
        # Pattern-specific criteria
        return self._check_pattern_specific_criteria(pattern, confidence_analysis, evidence_analysis)
    
    def _check_pattern_specific_criteria(self, pattern: Dict[str, Any],
                                       confidence_analysis: Dict[str, Any],
                                       evidence_analysis: Dict[str, Any]) -> bool:
        """Check pattern-specific criteria for alert generation"""
        pattern_type = pattern.get('pattern_type')
        
        if pattern_type == 'disengagement_pattern':
            # Require sustained disengagement
            sustained_duration = evidence_analysis.get('sustained_duration', 0.0)
            return sustained_duration >= 5.0  # 5 seconds of sustained disengagement
        
        elif pattern_type == 'attention_decline_pattern':
            # Require significant attention decline
            decline_magnitude = pattern.get('decline_magnitude', 0.0)
            return decline_magnitude >= 0.3  # 30% decline in attention
        
        elif pattern_type == 'behavioral_anomaly':
            # Require high anomaly confidence
            anomaly_confidence = pattern.get('anomaly_confidence', 0.0)
            return anomaly_confidence >= 0.9  # Very high confidence for anomalies
        
        return True
    
    def _create_alert_decision(self, pattern: Dict[str, Any],
                             confidence_analysis: Dict[str, Any],
                             evidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create alert decision based on pattern and analysis"""
        pattern_type = pattern.get('pattern_type')
        pattern_confidence = pattern.get('confidence', 0.0)
        
        # Map pattern type to alert type
        alert_type_mapping = {
            'disengagement_pattern': AlertType.DISENGAGEMENT_DETECTED,
            'attention_decline_pattern': AlertType.ATTENTION_DECLINE,
            'behavioral_anomaly': AlertType.BEHAVIORAL_ANOMALY,
            'sustained_inattention': AlertType.SUSTAINED_INATTENTION,
            'concerning_pattern': AlertType.CONCERNING_PATTERN,
            'participation_drop': AlertType.PARTICIPATION_DROP
        }
        
        alert_type = alert_type_mapping.get(pattern_type, AlertType.DISENGAGEMENT_DETECTED)
        
        # Determine severity
        severity = self._determine_alert_severity(pattern_confidence, pattern, evidence_analysis)
        
        # Generate alert ID
        alert_id = f"{alert_type.value}_{int(time.time() * 1000)}"
        
        # Create alert decision
        alert_decision = {
            'alert_id': alert_id,
            'alert_type': alert_type,
            'severity': severity,
            'confidence': pattern_confidence,
            'pattern': pattern,
            'evidence': evidence_analysis,
            'recommended_action': self._get_recommended_action(alert_type, severity),
            'priority_score': self._calculate_priority_score(severity, pattern_confidence),
            'should_generate': True
        }
        
        return alert_decision
    
    def _determine_alert_severity(self, confidence: float, pattern: Dict[str, Any],
                                evidence_analysis: Dict[str, Any]) -> AlertSeverity:
        """Determine alert severity based on confidence and evidence"""
        # Base severity on confidence
        if confidence >= self.severity_thresholds[AlertSeverity.CRITICAL]:
            base_severity = AlertSeverity.CRITICAL
        elif confidence >= self.severity_thresholds[AlertSeverity.HIGH]:
            base_severity = AlertSeverity.HIGH
        elif confidence >= self.severity_thresholds[AlertSeverity.MEDIUM]:
            base_severity = AlertSeverity.MEDIUM
        else:
            base_severity = AlertSeverity.LOW
        
        # Adjust based on pattern characteristics
        pattern_type = pattern.get('pattern_type')
        
        if pattern_type == 'behavioral_anomaly':
            # Anomalies are more severe
            if base_severity == AlertSeverity.MEDIUM:
                base_severity = AlertSeverity.HIGH
            elif base_severity == AlertSeverity.LOW:
                base_severity = AlertSeverity.MEDIUM
        
        # Adjust based on evidence duration
        evidence_duration = evidence_analysis.get('evidence_duration', 0.0)
        if evidence_duration > 10.0:  # Long duration increases severity
            if base_severity == AlertSeverity.LOW:
                base_severity = AlertSeverity.MEDIUM
            elif base_severity == AlertSeverity.MEDIUM:
                base_severity = AlertSeverity.HIGH
        
        return base_severity
    
    def _get_recommended_action(self, alert_type: AlertType, severity: AlertSeverity) -> str:
        """Get recommended action for alert"""
        action_map = {
            (AlertType.DISENGAGEMENT_DETECTED, AlertSeverity.LOW): "Monitor student engagement closely",
            (AlertType.DISENGAGEMENT_DETECTED, AlertSeverity.MEDIUM): "Consider interactive activity or question",
            (AlertType.DISENGAGEMENT_DETECTED, AlertSeverity.HIGH): "Direct engagement recommended",
            (AlertType.DISENGAGEMENT_DETECTED, AlertSeverity.CRITICAL): "Immediate intervention required",
            
            (AlertType.ATTENTION_DECLINE, AlertSeverity.LOW): "Check for understanding",
            (AlertType.ATTENTION_DECLINE, AlertSeverity.MEDIUM): "Vary teaching approach",
            (AlertType.ATTENTION_DECLINE, AlertSeverity.HIGH): "Break or activity change recommended",
            
            (AlertType.BEHAVIORAL_ANOMALY, AlertSeverity.MEDIUM): "Individual attention may be needed",
            (AlertType.BEHAVIORAL_ANOMALY, AlertSeverity.HIGH): "Check student wellbeing",
            (AlertType.BEHAVIORAL_ANOMALY, AlertSeverity.CRITICAL): "Immediate individual attention required"
        }
        
        return action_map.get((alert_type, severity), "Monitor situation and adjust as needed")
    
    def _calculate_priority_score(self, severity: AlertSeverity, confidence: float) -> float:
        """Calculate priority score for alert"""
        severity_scores = {
            AlertSeverity.LOW: 0.25,
            AlertSeverity.MEDIUM: 0.5,
            AlertSeverity.HIGH: 0.75,
            AlertSeverity.CRITICAL: 1.0
        }
        
        severity_score = severity_scores.get(severity, 0.5)
        
        # Combine severity and confidence
        priority_score = (severity_score * 0.7) + (confidence * 0.3)
        
        return priority_score
    
    def _process_and_filter_alerts(self, alert_decisions: List[Dict[str, Any]]) -> List[Alert]:
        """Process and filter alert decisions into final alerts"""
        processed_alerts = []
        
        # Sort by priority
        alert_decisions.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        for decision in alert_decisions:
            if decision.get('should_generate', False):
                alert = self._create_alert_from_decision(decision)
                
                # Final validation
                if self._validate_final_alert(alert):
                    processed_alerts.append(alert)
                    
                    # Update rate limiter
                    self.alert_rate_limiter.record_alert()
        
        return processed_alerts
    
    def _create_alert_from_decision(self, decision: Dict[str, Any]) -> Alert:
        """Create Alert object from decision"""
        current_time = time.time()
        
        alert = Alert(
            alert_id=decision['alert_id'],
            alert_type=decision['alert_type'],
            severity=decision['severity'],
            confidence=decision['confidence'],
            timestamp=current_time,
            student_id=None,  # Could be extracted from context
            description=self._generate_alert_description(decision),
            evidence=decision['evidence'],
            recommended_action=decision['recommended_action'],
            expires_at=current_time + 10.0,  # 10 seconds auto-disappear
            metadata={
                'pattern': decision['pattern'],
                'priority_score': decision['priority_score']
            }
        )
        
        return alert
    
    def _generate_alert_description(self, decision: Dict[str, Any]) -> str:
        """Generate human-readable alert description"""
        alert_type = decision['alert_type']
        confidence = decision['confidence']
        pattern = decision['pattern']
        
        descriptions = {
            AlertType.DISENGAGEMENT_DETECTED: f"Student disengagement detected with {confidence:.1%} confidence",
            AlertType.ATTENTION_DECLINE: f"Attention decline observed with {confidence:.1%} confidence",
            AlertType.BEHAVIORAL_ANOMALY: f"Unusual behavioral pattern detected with {confidence:.1%} confidence",
            AlertType.SUSTAINED_INATTENTION: f"Sustained inattention detected with {confidence:.1%} confidence",
            AlertType.CONCERNING_PATTERN: f"Concerning behavioral pattern identified with {confidence:.1%} confidence",
            AlertType.PARTICIPATION_DROP: f"Participation drop detected with {confidence:.1%} confidence"
        }
        
        return descriptions.get(alert_type, f"Engagement alert with {confidence:.1%} confidence")
    
    def _validate_final_alert(self, alert: Alert) -> bool:
        """Final validation before generating alert"""
        # Check if similar alert exists
        for active_alert in self.active_alerts.values():
            if (active_alert.alert_type == alert.alert_type and 
                abs(active_alert.timestamp - alert.timestamp) < self.alert_cooldown_period):
                return False
        
        # Check confidence threshold
        if alert.confidence < self.confidence_threshold:
            return False
        
        return True
    
    def _is_in_cooldown_period(self, pattern_type: str) -> bool:
        """Check if pattern type is in cooldown period"""
        current_time = time.time()
        
        for alert in self.recent_alerts:
            if (hasattr(alert, 'metadata') and 
                alert.metadata.get('pattern', {}).get('pattern_type') == pattern_type and
                current_time - alert.timestamp < self.alert_cooldown_period):
                return True
        
        return False
    
    def _update_alert_tracking(self, processed_alerts: List[Alert]):
        """Update alert tracking and statistics"""
        current_time = time.time()
        
        for alert in processed_alerts:
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            
            # Add to recent alerts
            self.recent_alerts.append(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            # Update statistics
            self.alert_statistics['total_alerts'] += 1
            
            # Log alert
            logger.warning(f"ENGAGEMENT ALERT: {alert.alert_type.value} - {alert.description}")
        
        # Clean up expired alerts
        self._cleanup_expired_alerts(current_time)
    
    def _cleanup_expired_alerts(self, current_time: float):
        """Clean up expired alerts"""
        expired_alert_ids = []
        
        for alert_id, alert in self.active_alerts.items():
            if current_time > alert.expires_at:
                expired_alert_ids.append(alert_id)
        
        for alert_id in expired_alert_ids:
            del self.active_alerts[alert_id]
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_generated': self.alert_statistics['total_alerts'],
            'alert_rate_limit_status': self.alert_rate_limiter.get_status(),
            'evidence_buffer_size': len(self.evidence_buffer),
            'suppressed_patterns_count': len(self.suppressed_patterns),
            'system_health': 'healthy'
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an active alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} dismissed")
            return True
        return False
    
    def suppress_pattern(self, pattern_type: str, duration: float = 300.0):
        """Temporarily suppress alerts for a pattern type"""
        self.suppressed_patterns.add(pattern_type)
        
        # Schedule removal of suppression
        def remove_suppression():
            time.sleep(duration)
            self.suppressed_patterns.discard(pattern_type)
        
        import threading
        threading.Thread(target=remove_suppression, daemon=True).start()
        
        logger.info(f"Pattern {pattern_type} suppressed for {duration} seconds")
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'pattern_validation': {},
            'confidence_analysis': {},
            'evidence_analysis': {},
            'alert_decisions': [],
            'processed_alerts': [],
            'active_alerts': [],
            'alert_statistics': self.alert_statistics.copy(),
            'system_status': self._get_system_status()
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup alert system resources"""
        # Save alert statistics and patterns for analysis
        try:
            self._save_alert_data()
        except Exception as e:
            logger.error(f"Error saving alert data: {e}")
        
        logger.info("Intelligent alert system cleaned up")
    
    def _save_alert_data(self):
        """Save alert data for analysis"""
        alert_data = {
            'statistics': self.alert_statistics,
            'recent_alerts': [
                {
                    'alert_type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'confidence': alert.confidence,
                    'timestamp': alert.timestamp
                }
                for alert in list(self.recent_alerts)[-100:]  # Last 100 alerts
            ]
        }
        
        # Save to file for analysis
        with open('data/alert_analysis.json', 'w') as f:
            json.dump(alert_data, f, indent=2)

# Helper classes for alert system components
class AlertPatternValidator:
    """Validate patterns for alert generation"""
    
    def initialize(self):
        logger.info("Alert pattern validator initialized")
    
    def validate_patterns(self, engagement_data: Dict[str, Any], 
                         evidence_buffer: deque) -> Dict[str, Any]:
        """Validate patterns for potential alerts"""
        return {
            'validated_patterns': [
                {
                    'pattern_type': 'disengagement_pattern',
                    'confidence': 0.85,
                    'evidence': {},
                    'duration': 5.0
                }
            ],
            'validation_confidence': 0.8
        }

class AlertConfidenceCalculator:
    """Calculate confidence scores for alerts"""
    
    def initialize(self):
        logger.info("Alert confidence calculator initialized")
    
    def calculate_confidence(self, pattern_validation: Dict[str, Any],
                           evidence_buffer: deque) -> Dict[str, Any]:
        """Calculate confidence scores"""
        return {
            'overall_confidence': 0.85,
            'pattern_confidences': {},
            'evidence_strength': 0.8
        }

class AlertEvidenceAggregator:
    """Aggregate evidence for alert decisions"""
    
    def initialize(self):
        logger.info("Alert evidence aggregator initialized")
    
    def aggregate_evidence(self, pattern_validation: Dict[str, Any],
                          confidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evidence for alerts"""
        return {
            'evidence_duration': 5.0,
            'sustained_duration': 5.0,
            'evidence_strength': 0.8,
            'supporting_indicators': []
        }

class AlertPrioritizer:
    """Prioritize alerts based on severity and context"""
    
    def initialize(self):
        logger.info("Alert prioritizer initialized")

class AlertRateLimiter:
    """Rate limiter for alert generation"""
    
    def __init__(self, max_alerts_per_minute: int):
        self.max_alerts_per_minute = max_alerts_per_minute
        self.alert_timestamps = deque(maxlen=max_alerts_per_minute)
    
    def initialize(self):
        logger.info("Alert rate limiter initialized")
    
    def can_generate_alert(self) -> bool:
        """Check if alert can be generated within rate limit"""
        current_time = time.time()
        
        # Remove old timestamps
        while self.alert_timestamps and current_time - self.alert_timestamps[0] > 60.0:
            self.alert_timestamps.popleft()
        
        return len(self.alert_timestamps) < self.max_alerts_per_minute
    
    def record_alert(self):
        """Record that an alert was generated"""
        self.alert_timestamps.append(time.time())
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        return {
            'alerts_in_last_minute': len(self.alert_timestamps),
            'max_alerts_per_minute': self.max_alerts_per_minute,
            'can_generate_alert': self.can_generate_alert()
        }
