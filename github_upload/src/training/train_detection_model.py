"""
Training Script for Advanced Human Detection Model
Uses state-of-the-art datasets and techniques for superior human detection
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.detection_model_trainer import AdvancedDetectionTrainer, StateOfTheArtDatasetManager
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description='Train advanced human detection model')
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--datasets', nargs='+', default=['coco_person'],
                       choices=['coco_person', 'crowdhuman', 'citypersons', 'wider_person', 'human_parts'],
                       help='Datasets to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--model-size', type=str, default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='YOLOv8 model size')
    parser.add_argument('--input-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare datasets without training')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--export-format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'tflite', 'tensorrt'],
                       help='Export format for optimized model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'model_size': args.model_size,
        'input_size': args.input_size,
        'learning_rate': args.learning_rate,
        'use_wandb': args.use_wandb
    })
    
    logger.info("🚀 Starting Advanced Human Detection Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    logger.info(f"Datasets to use: {args.datasets}")
    
    # Initialize trainer
    trainer = AdvancedDetectionTrainer(config)
    
    # Prepare datasets
    logger.info("📊 Preparing state-of-the-art datasets...")
    success = trainer.prepare_datasets(args.datasets)
    if not success:
        logger.error("Failed to prepare datasets")
        return 1
    
    if args.prepare_only:
        logger.info("✅ Dataset preparation completed")
        return 0
    
    # Initialize model
    if not args.evaluate_only:
        logger.info("🤖 Initializing YOLOv8 model...")
        success = trainer.initialize_model()
        if not success:
            logger.error("Failed to initialize model")
            return 1
    
    # Training
    if not args.evaluate_only:
        logger.info("🏋️ Starting advanced training...")
        
        # Use the first dataset for training (can be extended to combine multiple)
        dataset_path = Path("data/training_datasets") / args.datasets[0] / "dataset.yaml"
        
        if not dataset_path.exists():
            logger.error(f"Dataset configuration not found: {dataset_path}")
            return 1
        
        success = trainer.train_with_advanced_techniques(str(dataset_path))
        if not success:
            logger.error("Training failed")
            return 1
        
        logger.info("✅ Training completed successfully")
    
    # Evaluation
    logger.info("📈 Evaluating model performance...")
    dataset_path = Path("data/training_datasets") / args.datasets[0] / "dataset.yaml"
    metrics = trainer.evaluate_model(str(dataset_path))
    
    if metrics:
        logger.info("📊 Final Model Performance:")
        logger.info(f"  🎯 mAP@0.5: {metrics['mAP50']:.4f}")
        logger.info(f"  🎯 mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
        logger.info(f"  🎯 Precision: {metrics['precision']:.4f}")
        logger.info(f"  🎯 Recall: {metrics['recall']:.4f}")
        logger.info(f"  🎯 F1-Score: {metrics['f1_score']:.4f}")
        
        # Save metrics
        metrics_file = Path("checkpoints/detection_training/final_metrics.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"📄 Metrics saved to: {metrics_file}")
    
    # Export optimized model
    logger.info(f"📦 Exporting model in {args.export_format} format...")
    export_path = trainer.export_optimized_model(args.export_format)
    if export_path:
        logger.info(f"✅ Optimized model exported to: {export_path}")
    
    logger.info("🎉 Advanced human detection training pipeline completed!")
    return 0

def load_config(config_path: str) -> dict:
    """Load training configuration from file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()

def get_default_config() -> dict:
    """Get default training configuration"""
    return {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "model_size": "yolov8n",
        "input_size": 640,
        "use_augmentation": True,
        "use_mixup": True,
        "use_mosaic": True,
        "use_focal_loss": True,
        "use_wandb": False,
        "checkpoint_dir": "checkpoints/detection_training"
    }

if __name__ == "__main__":
    exit(main())
