#!/usr/bin/env python3
"""
Demo Script for Advanced Training System
Demonstrates the complete training pipeline with feedback reinforcement
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.model_integration import ModelIntegrator
from src.modules.automated_attendance_system import AutomatedAttendanceSystem
from src.utils.logger import logger

def demo_training_pipeline():
    """Demonstrate the complete training and feedback pipeline"""
    
    logger.info("🚀 DEMO: Advanced Training System with Feedback Reinforcement")
    logger.info("=" * 80)
    
    # 1. Initialize Model Integrator
    logger.info("📦 Step 1: Initialize Model Management System")
    integrator = ModelIntegrator({})
    
    # List existing models
    models = integrator.list_models()
    logger.info(f"Found {len(models)} existing models:")
    for model in models:
        logger.info(f"  📋 {model['model_id']}: {model['model_name']} "
                   f"(mAP50: {model['performance'].get('mAP50', 0):.4f})")
    
    # 2. Initialize Attendance System with Feedback
    logger.info("\n🤖 Step 2: Initialize Attendance System with Feedback Learning")
    config = {
        'face_detection_confidence': 0.03,
        'face_recognition_threshold': 0.7,
        'use_trained_model': True,
        'enable_feedback_training': True
    }
    
    attendance_system = AutomatedAttendanceSystem(config)
    
    # 3. Demonstrate Feedback Collection
    logger.info("\n📊 Step 3: Demonstrate Feedback Reinforcement Learning")
    
    # Simulate detection feedback scenarios
    feedback_scenarios = [
        {
            'type': 'correct_detection',
            'data': {
                'detection_method': 'face_detection',
                'confidence': 0.85,
                'person_id': 'C001'
            },
            'description': 'Correctly detected known student'
        },
        {
            'type': 'false_positive',
            'data': {
                'detection_method': 'face_detection',
                'confidence': 0.45,
                'person_id': 'unknown'
            },
            'description': 'False positive detection (object mistaken for person)'
        },
        {
            'type': 'missed_person',
            'data': {
                'detection_method': 'body_detection',
                'confidence': 0.25,
                'person_id': 'C002'
            },
            'description': 'Missed detecting a person in the background'
        },
        {
            'type': 'correct_recognition',
            'data': {
                'detection_method': 'face_recognition',
                'confidence': 0.92,
                'person_id': 'C003'
            },
            'description': 'Correctly recognized student face'
        },
        {
            'type': 'incorrect_recognition',
            'data': {
                'detection_method': 'face_recognition',
                'confidence': 0.78,
                'person_id': 'C001',
                'actual_person': 'C002'
            },
            'description': 'Incorrectly identified student (confused two people)'
        }
    ]
    
    logger.info("Simulating feedback scenarios...")
    for i, scenario in enumerate(feedback_scenarios, 1):
        logger.info(f"\n  📝 Scenario {i}: {scenario['description']}")
        
        # Add feedback to system
        attendance_system.add_detection_feedback(
            feedback_type=scenario['type'],
            detection_data=scenario['data']
        )
        
        # Show threshold adjustments
        summary = attendance_system.get_feedback_summary()
        logger.info(f"     🎯 Current Thresholds:")
        logger.info(f"       Face: {summary['current_thresholds']['face_threshold']:.3f}")
        logger.info(f"       Body: {summary['current_thresholds']['body_threshold']:.3f}")
        logger.info(f"       Recognition: {summary['current_thresholds']['recognition_threshold']:.3f}")
        
        time.sleep(1)  # Pause for demonstration
    
    # 4. Show Learning Progress
    logger.info("\n📈 Step 4: Learning Progress Summary")
    final_summary = attendance_system.get_feedback_summary()
    
    logger.info(f"Total Feedback Instances: {final_summary['total_feedback_instances']}")
    logger.info(f"Positive Feedback: {final_summary['positive_feedback_count']}")
    logger.info(f"Negative Feedback: {final_summary['negative_feedback_count']}")
    logger.info(f"Feedback Ratio: {final_summary['feedback_ratio']:.2%}")
    logger.info(f"Recent Accuracy: {final_summary['recent_accuracy']:.2%}")
    logger.info(f"Learning Rate: {final_summary['learning_rate']:.4f}")
    
    # 5. Demonstrate Model Comparison
    if len(models) > 1:
        logger.info("\n🏆 Step 5: Model Performance Comparison")
        model_ids = [m['model_id'] for m in models[:3]]
        comparison = integrator.compare_models(model_ids)
        
        if comparison and 'models' in comparison:
            logger.info("Model Performance Comparison:")
            for model_id, metrics in comparison['models'].items():
                logger.info(f"  📊 {metrics['model_name']}:")
                logger.info(f"     mAP@0.5: {metrics['mAP50']:.4f}")
                logger.info(f"     Precision: {metrics['precision']:.4f}")
                logger.info(f"     Recall: {metrics['recall']:.4f}")
                logger.info(f"     Size: {metrics['model_size_mb']:.1f} MB")
            
            if comparison.get('best_model'):
                logger.info(f"🥇 Best Model: {comparison['best_model']}")
    
    # 6. Show Active Model Info
    logger.info("\n🎯 Step 6: Active Model Information")
    active_model = integrator.get_active_model_info()
    if active_model:
        logger.info(f"Active Model: {active_model['model_name']}")
        logger.info(f"Performance: mAP50={active_model['performance_metrics'].get('mAP50', 0):.4f}")
        logger.info(f"Registration Date: {active_model['registration_date']}")
        logger.info(f"Model Size: {active_model['model_size'] / (1024*1024):.1f} MB")
    else:
        logger.info("No active model deployed")
    
    # 7. Training Recommendations
    logger.info("\n💡 Step 7: Training Recommendations")
    
    if final_summary['feedback_ratio'] < 0.7:
        logger.info("🔄 RECOMMENDATION: Model needs retraining")
        logger.info("   - Low positive feedback ratio detected")
        logger.info("   - Consider training with more diverse datasets")
        logger.info("   - Run: python scripts/train_and_deploy_model.py --datasets coco_person crowdhuman")
    
    if final_summary['recent_accuracy'] < 0.8:
        logger.info("⚠️  RECOMMENDATION: Increase training epochs")
        logger.info("   - Recent accuracy below 80%")
        logger.info("   - Consider longer training with more epochs")
    
    if not active_model or active_model['performance_metrics'].get('mAP50', 0) < 0.85:
        logger.info("🚀 RECOMMENDATION: Deploy better model")
        logger.info("   - Current model performance below optimal")
        logger.info("   - Train larger model: --model-size yolov8m")
    
    logger.info("\n✅ Demo completed successfully!")
    logger.info("🎉 The system demonstrates:")
    logger.info("   ✓ Real-time feedback learning")
    logger.info("   ✓ Adaptive threshold adjustment")
    logger.info("   ✓ Model performance tracking")
    logger.info("   ✓ Intelligent training recommendations")
    logger.info("   ✓ State-of-the-art model integration")

def demo_quick_training():
    """Demonstrate quick training with small dataset"""
    logger.info("\n🏃 QUICK TRAINING DEMO")
    logger.info("This would normally run:")
    logger.info("python scripts/train_and_deploy_model.py \\")
    logger.info("    --datasets coco_person \\")
    logger.info("    --epochs 50 \\")
    logger.info("    --batch-size 16 \\")
    logger.info("    --model-size yolov8s \\")
    logger.info("    --model-name 'demo_detector' \\")
    logger.info("    --deploy")
    
    logger.info("\n📊 Expected Results:")
    logger.info("   - Training Time: ~2-4 hours (depending on hardware)")
    logger.info("   - Expected mAP@0.5: 0.85+ (COCO Person)")
    logger.info("   - Model Size: ~22 MB (YOLOv8s)")
    logger.info("   - Inference Speed: 80+ FPS")

if __name__ == "__main__":
    try:
        demo_training_pipeline()
        demo_quick_training()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        sys.exit(1)
