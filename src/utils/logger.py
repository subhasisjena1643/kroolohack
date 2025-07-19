"""
Logging utility for Real-time Classroom Engagement Analyzer
"""

import logging
import os
from datetime import datetime
from typing import Optional

class EngagementLogger:
    """Custom logger for the engagement analyzer"""
    
    def __init__(self, name: str = "EngagementAnalyzer", log_file: Optional[str] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def performance(self, operation: str, duration: float, details: str = ""):
        """Log performance metrics"""
        message = f"PERFORMANCE - {operation}: {duration:.3f}s"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def engagement_event(self, event_type: str, data: dict):
        """Log engagement-specific events"""
        message = f"ENGAGEMENT - {event_type}: {data}"
        self.logger.info(message)

# Global logger instance
logger = EngagementLogger()
