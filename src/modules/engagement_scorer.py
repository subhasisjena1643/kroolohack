"""
Engagement Scoring Algorithm
Combines all detection metrics to calculate real-time engagement scores
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import statistics

from src.utils.base_processor import BaseProcessor
from src.utils.logger import logger

class EngagementScorer(BaseProcessor):
    """Real-time engagement scoring combining all AI modules"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EngagementScorer", config)
        
        # Scoring weights from config
        self.attention_weight = config.get('attention_weight', 0.3)
        self.participation_weight = config.get('participation_weight', 0.25)
        self.audio_engagement_weight = config.get('audio_engagement_weight', 0.25)
        self.posture_weight = config.get('posture_weight', 0.2)
        
        # Engagement thresholds
        self.high_engagement_threshold = config.get('high_engagement_threshold', 0.7)
        self.medium_engagement_threshold = config.get('medium_engagement_threshold', 0.4)
        
        # History and tracking
        self.history_window = config.get('history_window', 30)  # seconds
        self.engagement_history = deque(maxlen=self.history_window * 10)  # 10 FPS
        self.individual_scores = {}
        
        # Current metrics
        self.current_engagement_score = 0.0
        self.current_engagement_level = 'low'
        self.engagement_trends = {}
        
        # Analytics
        self.session_stats = {
            'start_time': time.time(),
            'total_high_engagement_time': 0.0,
            'total_medium_engagement_time': 0.0,
            'total_low_engagement_time': 0.0,
            'peak_engagement': 0.0,
            'average_engagement': 0.0
        }
    
    def initialize(self) -> bool:
        """Initialize engagement scorer"""
        try:
            logger.info("Initializing engagement scorer...")
            
            # Validate weights sum to 1.0
            total_weight = (self.attention_weight + self.participation_weight + 
                          self.audio_engagement_weight + self.posture_weight)
            
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Engagement weights sum to {total_weight}, normalizing...")
                # Normalize weights
                self.attention_weight /= total_weight
                self.participation_weight /= total_weight
                self.audio_engagement_weight /= total_weight
                self.posture_weight /= total_weight
            
            logger.info(f"Engagement weights - Attention: {self.attention_weight:.2f}, "
                       f"Participation: {self.participation_weight:.2f}, "
                       f"Audio: {self.audio_engagement_weight:.2f}, "
                       f"Posture: {self.posture_weight:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize engagement scorer: {e}")
            return False
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process combined data from all modules to calculate engagement"""
        try:
            # Extract data from different modules
            face_data = data.get('face_detection', {})
            pose_data = data.get('pose_estimation', {})
            gesture_data = data.get('gesture_recognition', {})
            audio_data = data.get('audio_processing', {})
            
            # Calculate individual component scores
            attention_score = self._calculate_attention_score(face_data, pose_data)
            participation_score = self._calculate_participation_score(gesture_data)
            audio_engagement_score = self._calculate_audio_engagement_score(audio_data)
            posture_score = self._calculate_posture_score(pose_data)
            
            # Calculate overall engagement score
            overall_engagement = (
                attention_score * self.attention_weight +
                participation_score * self.participation_weight +
                audio_engagement_score * self.audio_engagement_weight +
                posture_score * self.posture_weight
            )
            
            # Determine engagement level
            engagement_level = self._determine_engagement_level(overall_engagement)
            
            # Update history and trends
            self._update_engagement_history(overall_engagement, engagement_level)
            
            # Calculate trends and analytics
            trends = self._calculate_trends()
            analytics = self._calculate_analytics()
            
            # Update session statistics
            self._update_session_stats(overall_engagement, engagement_level)
            
            # Create detailed result
            result = {
                'overall_engagement_score': overall_engagement,
                'engagement_level': engagement_level,
                'component_scores': {
                    'attention': attention_score,
                    'participation': participation_score,
                    'audio_engagement': audio_engagement_score,
                    'posture': posture_score
                },
                'component_weights': {
                    'attention': self.attention_weight,
                    'participation': self.participation_weight,
                    'audio_engagement': self.audio_engagement_weight,
                    'posture': self.posture_weight
                },
                'trends': trends,
                'analytics': analytics,
                'session_stats': self.session_stats.copy(),
                'individual_metrics': self._get_individual_metrics(data),
                'recommendations': self._generate_recommendations(overall_engagement, trends)
            }
            
            # Update current state
            self.current_engagement_score = overall_engagement
            self.current_engagement_level = engagement_level
            
            return result
            
        except Exception as e:
            logger.error(f"Error in engagement scoring: {e}")
            return self._empty_result(error=str(e))
    
    def _calculate_attention_score(self, face_data: Dict[str, Any], pose_data: Dict[str, Any]) -> float:
        """Calculate attention score from face detection and pose estimation"""
        try:
            attention_score = 0.0
            
            # Face detection contribution (presence and count)
            face_count = face_data.get('face_count', 0)
            attendance_count = face_data.get('attendance_count', 1)
            
            if attendance_count > 0:
                presence_ratio = min(1.0, face_count / attendance_count)
                attention_score += presence_ratio * 0.4  # 40% from presence
            
            # Head pose contribution (looking at screen)
            pose_attention = pose_data.get('average_attention', 0.0)
            attention_score += pose_attention * 0.6  # 60% from head pose
            
            return min(1.0, attention_score)
            
        except Exception as e:
            logger.error(f"Error calculating attention score: {e}")
            return 0.0
    
    def _calculate_participation_score(self, gesture_data: Dict[str, Any]) -> float:
        """Calculate participation score from gesture recognition"""
        try:
            # Get participation score from gesture module
            participation_score = gesture_data.get('participation_score', 0.0)
            
            # Get recent participation events
            recent_events = gesture_data.get('participation_events', [])
            
            # Boost score based on recent activity
            if recent_events:
                recent_boost = min(0.3, len(recent_events) * 0.1)
                participation_score += recent_boost
            
            return min(1.0, participation_score)
            
        except Exception as e:
            logger.error(f"Error calculating participation score: {e}")
            return 0.0
    
    def _calculate_audio_engagement_score(self, audio_data: Dict[str, Any]) -> float:
        """Calculate audio engagement score"""
        try:
            engagement_metrics = audio_data.get('engagement_metrics', {})
            return engagement_metrics.get('audio_engagement_score', 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating audio engagement score: {e}")
            return 0.0
    
    def _calculate_posture_score(self, pose_data: Dict[str, Any]) -> float:
        """Calculate posture score from pose estimation"""
        try:
            # Use attention distribution as proxy for posture
            attention_dist = pose_data.get('attention_distribution', {})
            
            # Good posture = high attention states
            high_attention = attention_dist.get('high', 0.0)
            medium_attention = attention_dist.get('medium', 0.0)
            
            posture_score = high_attention * 1.0 + medium_attention * 0.6
            
            return min(1.0, posture_score)
            
        except Exception as e:
            logger.error(f"Error calculating posture score: {e}")
            return 0.0
    
    def _determine_engagement_level(self, score: float) -> str:
        """Determine engagement level from score"""
        if score >= self.high_engagement_threshold:
            return 'high'
        elif score >= self.medium_engagement_threshold:
            return 'medium'
        else:
            return 'low'
    
    def _update_engagement_history(self, score: float, level: str):
        """Update engagement history"""
        current_time = time.time()
        
        self.engagement_history.append({
            'timestamp': current_time,
            'score': score,
            'level': level
        })
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate engagement trends"""
        if len(self.engagement_history) < 10:
            return {'trend': 'stable', 'change_rate': 0.0, 'confidence': 'low'}
        
        # Get recent scores
        recent_scores = [entry['score'] for entry in list(self.engagement_history)[-20:]]
        
        # Calculate trend
        if len(recent_scores) >= 10:
            early_avg = statistics.mean(recent_scores[:5])
            late_avg = statistics.mean(recent_scores[-5:])
            
            change_rate = (late_avg - early_avg) / early_avg if early_avg > 0 else 0.0
            
            if change_rate > 0.1:
                trend = 'increasing'
            elif change_rate < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Calculate confidence based on consistency
            score_variance = statistics.variance(recent_scores)
            confidence = 'high' if score_variance < 0.05 else 'medium' if score_variance < 0.15 else 'low'
        else:
            trend = 'stable'
            change_rate = 0.0
            confidence = 'low'
        
        return {
            'trend': trend,
            'change_rate': change_rate,
            'confidence': confidence,
            'recent_average': statistics.mean(recent_scores) if recent_scores else 0.0,
            'variance': statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.0
        }
    
    def _calculate_analytics(self) -> Dict[str, Any]:
        """Calculate detailed analytics"""
        if not self.engagement_history:
            return {}
        
        scores = [entry['score'] for entry in self.engagement_history]
        levels = [entry['level'] for entry in self.engagement_history]
        
        # Level distribution
        level_counts = {'high': 0, 'medium': 0, 'low': 0}
        for level in levels:
            level_counts[level] += 1
        
        total_entries = len(levels)
        level_distribution = {
            level: count / total_entries for level, count in level_counts.items()
        } if total_entries > 0 else {'high': 0, 'medium': 0, 'low': 0}
        
        # Score statistics
        score_stats = {
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'min': min(scores),
            'max': max(scores),
            'range': max(scores) - min(scores)
        }
        
        return {
            'level_distribution': level_distribution,
            'score_statistics': score_stats,
            'total_measurements': total_entries,
            'measurement_duration': self.engagement_history[-1]['timestamp'] - self.engagement_history[0]['timestamp']
        }
    
    def _update_session_stats(self, score: float, level: str):
        """Update session statistics"""
        current_time = time.time()
        
        # Update peak engagement
        if score > self.session_stats['peak_engagement']:
            self.session_stats['peak_engagement'] = score
        
        # Update time in each level (approximate)
        time_increment = 0.1  # Assuming 10 FPS processing
        
        if level == 'high':
            self.session_stats['total_high_engagement_time'] += time_increment
        elif level == 'medium':
            self.session_stats['total_medium_engagement_time'] += time_increment
        else:
            self.session_stats['total_low_engagement_time'] += time_increment
        
        # Update average engagement
        if self.engagement_history:
            scores = [entry['score'] for entry in self.engagement_history]
            self.session_stats['average_engagement'] = statistics.mean(scores)
    
    def _get_individual_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get individual metrics for detailed analysis"""
        return {
            'face_detection': {
                'face_count': data.get('face_detection', {}).get('face_count', 0),
                'attendance_count': data.get('face_detection', {}).get('attendance_count', 0)
            },
            'pose_estimation': {
                'average_attention': data.get('pose_estimation', {}).get('average_attention', 0.0),
                'attention_distribution': data.get('pose_estimation', {}).get('attention_distribution', {})
            },
            'gesture_recognition': {
                'participation_score': data.get('gesture_recognition', {}).get('participation_score', 0.0),
                'recent_events_count': len(data.get('gesture_recognition', {}).get('participation_events', []))
            },
            'audio_processing': {
                'speech_ratio': data.get('audio_processing', {}).get('engagement_metrics', {}).get('speech_ratio', 0.0),
                'sentiment_score': data.get('audio_processing', {}).get('engagement_metrics', {}).get('sentiment_score', 0.0),
                'active_speakers': data.get('audio_processing', {}).get('speaker_stats', {}).get('active_speakers', 0)
            }
        }
    
    def _generate_recommendations(self, score: float, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on engagement analysis"""
        recommendations = []
        
        # Score-based recommendations
        if score < 0.3:
            recommendations.append("Consider interactive activities to boost engagement")
            recommendations.append("Check if content difficulty is appropriate")
        elif score < 0.6:
            recommendations.append("Add more interactive elements")
            recommendations.append("Encourage participation through questions")
        
        # Trend-based recommendations
        if trends.get('trend') == 'decreasing':
            recommendations.append("Engagement is declining - consider a break or activity change")
            recommendations.append("Check for distractions or technical issues")
        elif trends.get('trend') == 'stable' and score < 0.5:
            recommendations.append("Try varying teaching methods to increase engagement")
        
        # Component-specific recommendations
        # These would be based on individual component scores
        # (Implementation would depend on having access to component scores)
        
        return recommendations
    
    def get_engagement_summary(self) -> Dict[str, Any]:
        """Get comprehensive engagement summary"""
        return {
            'current_score': self.current_engagement_score,
            'current_level': self.current_engagement_level,
            'session_duration': time.time() - self.session_stats['start_time'],
            'session_stats': self.session_stats.copy(),
            'recent_trend': self._calculate_trends(),
            'analytics': self._calculate_analytics()
        }
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'overall_engagement_score': 0.0,
            'engagement_level': 'low',
            'component_scores': {
                'attention': 0.0,
                'participation': 0.0,
                'audio_engagement': 0.0,
                'posture': 0.0
            },
            'trends': {'trend': 'stable', 'change_rate': 0.0},
            'analytics': {},
            'session_stats': self.session_stats.copy(),
            'recommendations': []
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Engagement scorer cleaned up")
