#!/usr/bin/env python3
"""
Complete Training and Deployment Pipeline
Trains a state-of-the-art human detection model and deploys it to the system
"""

import argparse
import sys
import json
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.train_detection_model import main as train_main
from src.training.model_integration import ModelIntegrator
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description='Complete training and deployment pipeline')
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='Training configuration file')
    parser.add_argument('--model-name', type=str, default='advanced_human_detector',
                       help='Name for the trained model')
    parser.add_argument('--description', type=str, default='State-of-the-art human detection model',
                       help='Model description')
    parser.add_argument('--datasets', nargs='+', default=['coco_person'],
                       help='Datasets to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--model-size', type=str, default='yolov8s',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='Model size')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only deploy existing model')
    parser.add_argument('--model-path', type=str,
                       help='Path to existing model (if skipping training)')
    parser.add_argument('--deploy', action='store_true',
                       help='Deploy model after training')
    parser.add_argument('--backup-current', action='store_true', default=True,
                       help='Backup current model before deployment')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing models')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Complete Training and Deployment Pipeline")
    
    # Initialize model integrator
    integrator = ModelIntegrator({})
    
    if args.validate_only:
        logger.info("🔍 Validating existing models...")
        models = integrator.list_models()
        
        if not models:
            logger.info("No models found in registry")
            return 0
        
        logger.info(f"Found {len(models)} registered models:")
        for model in models:
            logger.info(f"  📦 {model['model_id']}: {model['model_name']} "
                       f"(mAP50: {model['performance'].get('mAP50', 0):.4f})")
            
            validation = integrator.validate_model(model['model_id'])
            if validation['valid']:
                logger.info(f"    ✅ Validation: PASSED")
            else:
                logger.info(f"    ❌ Validation: FAILED - {validation['error']}")
        
        return 0
    
    trained_model_path = None
    training_config = {}
    performance_metrics = {}
    
    if not args.skip_training:
        logger.info("🏋️ Starting model training...")
        
        # Prepare training arguments
        training_args = [
            '--config', args.config,
            '--datasets'] + args.datasets + [
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--model-size', args.model_size,
            '--use-wandb'
        ]
        
        # Override sys.argv for training script
        original_argv = sys.argv
        sys.argv = ['train_detection_model.py'] + training_args
        
        try:
            # Run training
            from src.training.train_detection_model import main as train_main
            result = train_main()
            
            if result != 0:
                logger.error("Training failed")
                return 1
            
            # Find the trained model
            checkpoint_dir = Path("checkpoints/detection_training")
            model_files = list(checkpoint_dir.glob("**/best.pt"))
            
            if not model_files:
                model_files = list(checkpoint_dir.glob("**/*.pt"))
            
            if model_files:
                trained_model_path = str(model_files[0])  # Use the first/best model
                logger.info(f"✅ Training completed. Model saved to: {trained_model_path}")
            else:
                logger.error("No trained model found")
                return 1
            
            # Load training configuration
            try:
                with open(args.config, 'r') as f:
                    training_config = json.load(f)
            except:
                training_config = {}
            
            # Load performance metrics
            metrics_file = checkpoint_dir / "final_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        performance_metrics = json.load(f)
                except:
                    performance_metrics = {}
            
        finally:
            sys.argv = original_argv
    
    else:
        if not args.model_path:
            logger.error("Model path required when skipping training")
            return 1
        
        trained_model_path = args.model_path
        if not Path(trained_model_path).exists():
            logger.error(f"Model file not found: {trained_model_path}")
            return 1
        
        logger.info(f"Using existing model: {trained_model_path}")
    
    # Register the model
    logger.info("📝 Registering trained model...")
    success = integrator.register_trained_model(
        model_path=trained_model_path,
        model_name=args.model_name,
        training_config=training_config,
        performance_metrics=performance_metrics,
        description=args.description
    )
    
    if not success:
        logger.error("Failed to register model")
        return 1
    
    # Get the registered model ID
    models = integrator.list_models()
    if not models:
        logger.error("No models found after registration")
        return 1
    
    latest_model = models[0]  # Most recent model
    model_id = latest_model['model_id']
    
    logger.info(f"✅ Model registered with ID: {model_id}")
    
    # Deploy the model if requested
    if args.deploy:
        logger.info("🚀 Deploying model...")
        
        success = integrator.deploy_model(
            model_id=model_id,
            backup_current=args.backup_current
        )
        
        if success:
            logger.info("✅ Model deployed successfully")
            
            # Update face detector configuration
            integrator.update_face_detector_config(model_id)
            logger.info("✅ Face detector configuration updated")
            
            # Show active model info
            active_model = integrator.get_active_model_info()
            if active_model:
                logger.info("📊 Active Model Information:")
                logger.info(f"  🏷️  Name: {active_model['model_name']}")
                logger.info(f"  🎯 mAP@0.5: {active_model['performance_metrics'].get('mAP50', 0):.4f}")
                logger.info(f"  🎯 Precision: {active_model['performance_metrics'].get('precision', 0):.4f}")
                logger.info(f"  🎯 Recall: {active_model['performance_metrics'].get('recall', 0):.4f}")
                logger.info(f"  📦 Size: {active_model['model_size'] / (1024*1024):.1f} MB")
        else:
            logger.error("Failed to deploy model")
            return 1
    
    # Show model comparison if multiple models exist
    if len(models) > 1:
        logger.info("📊 Model Comparison:")
        model_ids = [m['model_id'] for m in models[:5]]  # Compare top 5 models
        comparison = integrator.compare_models(model_ids)
        
        if comparison and 'models' in comparison:
            for mid, metrics in comparison['models'].items():
                status = "🟢 ACTIVE" if mid == integrator.model_registry.get("active_model") else "⚪"
                logger.info(f"  {status} {metrics['model_name']}: "
                           f"mAP50={metrics['mAP50']:.4f}, "
                           f"Size={metrics['model_size_mb']:.1f}MB")
            
            if comparison.get('best_model'):
                logger.info(f"🏆 Best performing model: {comparison['best_model']}")
    
    logger.info("🎉 Training and deployment pipeline completed successfully!")
    logger.info("💡 The system is now using the state-of-the-art trained model for human detection")
    
    return 0

if __name__ == "__main__":
    exit(main())
