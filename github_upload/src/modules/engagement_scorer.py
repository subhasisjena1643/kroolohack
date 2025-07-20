"""
Engagement Scoring Algorithm
Combines all detection metrics to calculate real-time engagement scores
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import statistics

from utils.base_processor import BaseProcessor
from utils.logger import logger

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
        self.individual_scores = {}  # Track scores for each detected person
        self.person_trackers = {}  # Track individual people across frames
        self.next_person_id = 1  # Assign unique IDs to people
        
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

            # Track individual people and their scores
            body_data = data.get('advanced_body_detection', {})
            self._update_individual_tracking(face_data, pose_data, gesture_data, body_data)
            
            # Calculate overall engagement score with real-time responsiveness
            overall_engagement = (
                attention_score * self.attention_weight +
                participation_score * self.participation_weight +
                audio_engagement_score * self.audio_engagement_weight +
                posture_score * self.posture_weight
            )

            # Apply smoothing but keep responsiveness for real-time updates
            if hasattr(self, 'previous_engagement'):
                # Use lighter smoothing for more responsive updates
                overall_engagement = 0.3 * self.previous_engagement + 0.7 * overall_engagement

            self.previous_engagement = overall_engagement
            
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

            # Debug logging for all scores every few seconds
            current_time = time.time()
            if not hasattr(self, '_last_score_debug_time') or current_time - self._last_score_debug_time > 3.0:
                logger.info(f"📊 ALL SCORES: Overall={overall_engagement:.3f}, "
                           f"Attention={attention_score:.3f}, Participation={participation_score:.3f}, "
                           f"Audio={audio_engagement_score:.3f}, Posture={posture_score:.3f}, "
                           f"Level={engagement_level}")
                self._last_score_debug_time = current_time

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
        """Calculate participation score from gesture recognition with real-time responsiveness"""
        try:
            # Get base participation score from gesture module
            base_participation_score = gesture_data.get('participation_score', 0.0)

            # Get recent participation events for immediate boost
            recent_events = gesture_data.get('participation_events', [])

            # Calculate immediate activity boost (more responsive)
            immediate_activity = 0.0
            if recent_events:
                # Count very recent events (last 2 seconds) for immediate response
                current_time = time.time()
                very_recent_events = [e for e in recent_events
                                    if current_time - e.get('timestamp', 0) < 2.0]
                immediate_activity = min(0.5, len(very_recent_events) * 0.2)

            # Combine base score with immediate activity
            final_score = min(1.0, base_participation_score + immediate_activity)

            return final_score

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

    def _update_individual_tracking(self, face_data: Dict[str, Any], pose_data: Dict[str, Any],
                                   gesture_data: Dict[str, Any], body_data: Dict[str, Any]):
        """Track individual people and calculate their engagement scores"""
        try:
            faces = face_data.get('faces', [])
            current_time = time.time()

            # Update individual scores for each detected face
            for i, face in enumerate(faces):
                person_id = f"person_{i}"  # Simple ID based on detection order

                # Calculate individual scores for this person
                individual_attention = self._calculate_individual_attention(face, pose_data)
                individual_participation = self._calculate_individual_participation(face, gesture_data)
                individual_posture = self._calculate_individual_posture(face, body_data)

                # Store individual scores
                if person_id not in self.individual_scores:
                    self.individual_scores[person_id] = {
                        'attention_history': deque(maxlen=30),
                        'participation_history': deque(maxlen=30),
                        'posture_history': deque(maxlen=30),
                        'last_seen': current_time,
                        'face_info': face
                    }

                # Update histories
                self.individual_scores[person_id]['attention_history'].append(individual_attention)
                self.individual_scores[person_id]['participation_history'].append(individual_participation)
                self.individual_scores[person_id]['posture_history'].append(individual_posture)
                self.individual_scores[person_id]['last_seen'] = current_time
                self.individual_scores[person_id]['face_info'] = face

                # Log individual scores every few seconds
                if not hasattr(self, '_last_individual_log_time') or current_time - self._last_individual_log_time > 5.0:
                    avg_attention = sum(self.individual_scores[person_id]['attention_history']) / len(self.individual_scores[person_id]['attention_history'])
                    avg_participation = sum(self.individual_scores[person_id]['participation_history']) / len(self.individual_scores[person_id]['participation_history'])
                    avg_posture = sum(self.individual_scores[person_id]['posture_history']) / len(self.individual_scores[person_id]['posture_history'])

                    logger.info(f"👤 {person_id.upper()}: Attention={avg_attention:.3f}, "
                               f"Participation={avg_participation:.3f}, Posture={avg_posture:.3f}, "
                               f"Confidence={face.get('confidence', 0.0):.3f}")

            # Clean up old person data (not seen for 10 seconds)
            persons_to_remove = []
            for person_id, data in self.individual_scores.items():
                if current_time - data['last_seen'] > 10.0:
                    persons_to_remove.append(person_id)

            for person_id in persons_to_remove:
                del self.individual_scores[person_id]

            # Update log time
            if len(faces) > 0 and (not hasattr(self, '_last_individual_log_time') or current_time - self._last_individual_log_time > 5.0):
                self._last_individual_log_time = current_time

        except Exception as e:
            logger.error(f"Error in individual tracking: {e}")

    def _calculate_individual_attention(self, face: Dict[str, Any], pose_data: Dict[str, Any]) -> float:
        """Calculate attention score for individual person"""
        try:
            # Base attention from face detection confidence
            attention = face.get('confidence', 0.0) * 0.5

            # Add pose-based attention if available
            poses = pose_data.get('poses', [])
            if poses:
                # Find closest pose to this face
                face_center = face.get('center', [0, 0])
                closest_pose = None
                min_distance = float('inf')

                for pose in poses:
                    pose_center = pose.get('center', [0, 0])
                    distance = ((face_center[0] - pose_center[0])**2 + (face_center[1] - pose_center[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_pose = pose

                if closest_pose and min_distance < 100:  # Within reasonable distance
                    attention += closest_pose.get('attention_score', 0.0) * 0.5

            return min(1.0, attention)

        except Exception as e:
            logger.error(f"Error calculating individual attention: {e}")
            return 0.0

    def _calculate_individual_participation(self, face: Dict[str, Any], gesture_data: Dict[str, Any]) -> float:
        """Calculate participation score for individual person"""
        try:
            # Base participation from face size (closer = more engaged)
            face_area = face.get('area', 0)
            participation = min(0.3, face_area / 10000)  # Normalize face area

            # Add gesture-based participation if available
            gestures = gesture_data.get('gestures', {})
            if gestures:
                # Simple heuristic: if any gestures detected, add to participation
                gesture_score = min(0.7, len(gestures) * 0.2)
                participation += gesture_score

            return min(1.0, participation)

        except Exception as e:
            logger.error(f"Error calculating individual participation: {e}")
            return 0.0

    def _calculate_individual_posture(self, face: Dict[str, Any], body_data: Dict[str, Any]) -> float:
        """Calculate posture score for individual person"""
        try:
            # Base posture from face position (centered = better posture)
            face_center = face.get('center', [320, 240])  # Default center
            frame_center = [320, 240]  # Assume 640x480 frame

            # Distance from center (normalized)
            distance = ((face_center[0] - frame_center[0])**2 + (face_center[1] - frame_center[1])**2)**0.5
            max_distance = (320**2 + 240**2)**0.5
            posture = 1.0 - (distance / max_distance)

            return max(0.0, posture)

        except Exception as e:
            logger.error(f"Error calculating individual posture: {e}")
            return 0.0

    def get_individual_scores(self) -> Dict[str, Any]:
        """Get current individual scores for all tracked people"""
        result = {}
        current_time = time.time()

        for person_id, data in self.individual_scores.items():
            if current_time - data['last_seen'] < 5.0:  # Only include recently seen people
                attention_history = list(data['attention_history'])
                participation_history = list(data['participation_history'])
                posture_history = list(data['posture_history'])

                result[person_id] = {
                    'current_attention': attention_history[-1] if attention_history else 0.0,
                    'current_participation': participation_history[-1] if participation_history else 0.0,
                    'current_posture': posture_history[-1] if posture_history else 0.0,
                    'average_attention': sum(attention_history) / len(attention_history) if attention_history else 0.0,
                    'average_participation': sum(participation_history) / len(participation_history) if participation_history else 0.0,
                    'average_posture': sum(posture_history) / len(posture_history) if posture_history else 0.0,
                    'face_confidence': data['face_info'].get('confidence', 0.0),
                    'face_area': data['face_info'].get('area', 0),
                    'last_seen': data['last_seen']
                }

        return result
    
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
