#!/usr/bin/env python3
"""
Test all imports to identify issues
"""

import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

def test_import(module_name, import_statement):
    """Test a single import"""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: OK")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    """Test all imports"""
    print("🔍 Testing imports...")
    print("=" * 50)
    
    imports_to_test = [
        ("Config", "from config.config import config"),
        ("Logger", "from utils.logger import logger"),
        ("Communication", "from utils.communication import CommunicationManager"),
        ("Face Detector", "from modules.face_detector import FaceDetector"),
        ("Pose Estimator", "from modules.pose_estimator import HeadPoseEstimator"),
        ("Gesture Recognizer", "from modules.gesture_recognizer import GestureRecognizer"),
        ("Audio Processor", "from modules.audio_processor import AudioProcessor"),
        ("Engagement Scorer", "from modules.engagement_scorer import EngagementScorer"),
        ("Advanced Body Detector", "from modules.advanced_body_detector import AdvancedBodyDetector"),
        ("Advanced Eye Tracker", "from modules.advanced_eye_tracker import AdvancedEyeTracker"),
        ("Micro Expression Analyzer", "from modules.micro_expression_analyzer import MicroExpressionAnalyzer"),
        ("Intelligent Pattern Analyzer", "from modules.intelligent_pattern_analyzer import IntelligentPatternAnalyzer"),
        ("Behavioral Classifier", "from modules.behavioral_classifier import BehavioralPatternClassifier"),
        ("Intelligent Alert System", "from modules.intelligent_alert_system import IntelligentAlertSystem"),
        ("Continuous Learning System", "from modules.continuous_learning_system import ContinuousLearningSystem"),
        ("Feedback Interface", "from modules.feedback_interface import FeedbackInterface"),
    ]
    
    success_count = 0
    total_count = len(imports_to_test)
    
    for module_name, import_statement in imports_to_test:
        if test_import(module_name, import_statement):
            success_count += 1
    
    print("=" * 50)
    print(f"📊 Results: {success_count}/{total_count} imports successful")
    
    if success_count == total_count:
        print("🎉 All imports working! You can run the application.")
        print("🚀 Try: python run_app.py")
    else:
        print("⚠️ Some imports failed. Check the errors above.")
        
    return success_count == total_count

if __name__ == "__main__":
    main()
