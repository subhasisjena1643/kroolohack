"""
Configuration file for Real-time Classroom Engagement Analyzer
Hackathon Project Configuration
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VideoConfig:
    """Video processing configuration"""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    
    # Processing settings
    max_latency_ms: int = 5000  # 5 seconds max latency
    buffer_size: int = 10
    
    # Face detection
    face_confidence_threshold: float = 0.5
    max_faces: int = 50  # Maximum faces to detect in classroom
    
    # Pose estimation
    pose_confidence_threshold: float = 0.5
    attention_angle_threshold: float = 30.0  # degrees
    
    # Gesture recognition
    gesture_confidence_threshold: float = 0.7
    gesture_buffer_frames: int = 10

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    # Recording settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "int16"
    
    # Processing settings
    silence_threshold: float = 0.01
    min_speech_duration: float = 1.0  # seconds
    max_speech_duration: float = 30.0  # seconds
    
    # Sentiment analysis
    sentiment_window_size: int = 5  # seconds
    sentiment_update_interval: float = 2.0  # seconds

@dataclass
class EngagementConfig:
    """Engagement scoring configuration"""
    # Scoring weights
    attention_weight: float = 0.3
    participation_weight: float = 0.25
    audio_engagement_weight: float = 0.25
    posture_weight: float = 0.2
    
    # Thresholds
    high_engagement_threshold: float = 0.7
    medium_engagement_threshold: float = 0.4
    
    # Update intervals
    score_update_interval: float = 1.0  # seconds
    history_window: int = 30  # seconds

@dataclass
class CommunicationConfig:
    """Communication and API configuration"""
    # WebSocket settings
    websocket_url: str = "ws://localhost:3000/engagement"
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    
    # API settings
    api_base_url: str = "http://localhost:3000/api"
    api_timeout: float = 5.0
    
    # Data format
    send_interval: float = 1.0  # seconds
    batch_size: int = 10

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/engagement_analyzer.log"
    
    # Performance
    enable_gpu: bool = False  # Laptop mode - no GPU
    num_threads: int = 4
    memory_limit_mb: int = 2048
    
    # Privacy
    save_faces: bool = False  # Privacy compliant
    anonymize_data: bool = True
    data_retention_hours: int = 24

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.video = VideoConfig()
        self.audio = AudioConfig()
        self.engagement = EngagementConfig()
        self.communication = CommunicationConfig()
        self.system = SystemConfig()
        
        # Load environment variables
        self._load_env_variables()
    
    def _load_env_variables(self):
        """Load configuration from environment variables"""
        # WebSocket URL
        if os.getenv("WEBSOCKET_URL"):
            self.communication.websocket_url = os.getenv("WEBSOCKET_URL")
        
        # API URL
        if os.getenv("API_BASE_URL"):
            self.communication.api_base_url = os.getenv("API_BASE_URL")
        
        # Camera index
        if os.getenv("CAMERA_INDEX"):
            self.video.camera_index = int(os.getenv("CAMERA_INDEX"))
        
        # Log level
        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "video": self.video.__dict__,
            "audio": self.audio.__dict__,
            "engagement": self.engagement.__dict__,
            "communication": self.communication.__dict__,
            "system": self.system.__dict__
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

# Global configuration instance
config = Config()
