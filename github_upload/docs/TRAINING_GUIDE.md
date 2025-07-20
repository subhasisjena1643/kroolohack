# 🚀 Advanced Human Detection Training Guide

This guide covers the state-of-the-art training system for improving human detection accuracy using external datasets and advanced techniques.

## 🎯 Overview

Our training system uses:
- **State-of-the-art datasets**: COCO Person, CrowdHuman, CityPersons, WIDER Person
- **Advanced augmentation**: Albumentations, Mixup, Mosaic, CutMix
- **Modern optimization**: AdamW, Cosine scheduling, Mixed precision
- **Robust evaluation**: Multiple metrics, cross-validation
- **Experiment tracking**: Weights & Biases integration

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for datasets
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements
```bash
# Install training dependencies
pip install -r requirements_training.txt

# Install additional tools
pip install wandb  # For experiment tracking
```

## 🗂️ Dataset Preparation

### Automatic Dataset Download
```bash
# Download and prepare COCO Person dataset
python src/training/train_detection_model.py --datasets coco_person --prepare-only

# Download multiple datasets
python src/training/train_detection_model.py --datasets coco_person crowdhuman citypersons --prepare-only
```

### Manual Dataset Setup
For datasets requiring manual download:

1. **CrowdHuman Dataset**:
   ```bash
   # Visit: https://www.crowdhuman.org/download
   # Download train/val sets to: data/training_datasets/crowdhuman/
   ```

2. **CityPersons Dataset**:
   ```bash
   # Visit: https://github.com/cvgroup-njust/CityPersons
   # Follow their download instructions
   ```

## 🏋️ Training Process

### Quick Start Training
```bash
# Basic training with COCO Person dataset
python scripts/train_and_deploy_model.py \
    --datasets coco_person \
    --epochs 100 \
    --batch-size 16 \
    --model-size yolov8s \
    --deploy

# Advanced training with multiple datasets
python scripts/train_and_deploy_model.py \
    --datasets coco_person crowdhuman citypersons \
    --epochs 150 \
    --batch-size 32 \
    --model-size yolov8m \
    --model-name "advanced_human_detector_v2" \
    --description "Multi-dataset trained model" \
    --deploy
```

### Custom Configuration Training
```bash
# Use custom configuration file
python scripts/train_and_deploy_model.py \
    --config configs/training_config.json \
    --model-name "custom_detector" \
    --deploy
```

### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--datasets` | Datasets to use | `coco_person` | `coco_person`, `crowdhuman`, `citypersons`, `wider_person` |
| `--epochs` | Training epochs | `100` | `50-300` |
| `--batch-size` | Batch size | `16` | `8-64` (depends on GPU) |
| `--model-size` | YOLOv8 model size | `yolov8s` | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` |
| `--learning-rate` | Learning rate | `0.001` | `0.0001-0.01` |

## 📊 Model Performance Tracking

### Weights & Biases Integration
```bash
# Enable W&B tracking
python scripts/train_and_deploy_model.py \
    --datasets coco_person \
    --epochs 100 \
    --use-wandb

# View results at: https://wandb.ai/your-project
```

### Local Metrics
Training metrics are saved to:
- `checkpoints/detection_training/final_metrics.json`
- `checkpoints/detection_training/training_plots/`

## 🔧 Model Management

### List Available Models
```bash
python scripts/train_and_deploy_model.py --validate-only
```

### Deploy Existing Model
```bash
python scripts/train_and_deploy_model.py \
    --skip-training \
    --model-path "path/to/your/model.pt" \
    --model-name "existing_model" \
    --deploy
```

### Model Comparison
```python
from src.training.model_integration import ModelIntegrator

integrator = ModelIntegrator({})
models = integrator.list_models()
comparison = integrator.compare_models([m['model_id'] for m in models[:3]])
print(comparison)
```

## 🎛️ Advanced Configuration

### Training Configuration (`configs/training_config.json`)

Key sections:
- **Model Configuration**: Architecture, input size, classes
- **Training Parameters**: Epochs, batch size, optimization
- **Data Augmentation**: Geometric, photometric, advanced techniques
- **Loss Configuration**: Focal loss, label smoothing
- **Hardware Optimization**: Mixed precision, multi-GPU

### Custom Augmentation Pipeline
```python
import albumentations as A

# Create custom augmentation
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    # Add more augmentations...
])
```

## 📈 Performance Optimization

### GPU Optimization
```bash
# Enable mixed precision training
export CUDA_VISIBLE_DEVICES=0
python scripts/train_and_deploy_model.py \
    --batch-size 32 \
    --model-size yolov8m
```

### Multi-GPU Training
```bash
# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
python scripts/train_and_deploy_model.py \
    --batch-size 64 \
    --model-size yolov8l
```

### Memory Optimization
```bash
# Reduce batch size for limited VRAM
python scripts/train_and_deploy_model.py \
    --batch-size 8 \
    --model-size yolov8n
```

## 🧪 Evaluation and Testing

### Model Evaluation
```bash
# Evaluate trained model
python src/training/train_detection_model.py \
    --evaluate-only \
    --datasets coco_person
```

### Performance Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Benchmark Results
Expected performance on COCO Person validation:

| Model Size | mAP@0.5 | mAP@0.5:0.95 | Speed (FPS) | Size (MB) |
|------------|---------|--------------|-------------|-----------|
| YOLOv8n | 0.85+ | 0.65+ | 100+ | 6 |
| YOLOv8s | 0.88+ | 0.68+ | 80+ | 22 |
| YOLOv8m | 0.90+ | 0.70+ | 60+ | 50 |
| YOLOv8l | 0.92+ | 0.72+ | 45+ | 87 |
| YOLOv8x | 0.93+ | 0.74+ | 30+ | 136 |

## 🚀 Deployment Integration

### Automatic Deployment
```bash
# Train and automatically deploy
python scripts/train_and_deploy_model.py \
    --datasets coco_person \
    --epochs 100 \
    --deploy \
    --backup-current
```

### Manual Integration
```python
from src.training.model_integration import ModelIntegrator

# Register and deploy model
integrator = ModelIntegrator({})
integrator.register_trained_model(
    model_path="path/to/model.pt",
    model_name="my_detector",
    training_config={},
    performance_metrics={"mAP50": 0.89}
)

# Deploy to system
integrator.deploy_model("model_id")
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   --batch-size 8
   
   # Use smaller model
   --model-size yolov8n
   ```

2. **Dataset Download Fails**:
   ```bash
   # Manual download and extract to:
   data/training_datasets/dataset_name/
   ```

3. **Training Stalls**:
   ```bash
   # Reduce learning rate
   --learning-rate 0.0005
   
   # Enable gradient clipping
   # (configured in training_config.json)
   ```

4. **Poor Performance**:
   - Increase training epochs
   - Use larger model size
   - Add more diverse datasets
   - Tune augmentation parameters

### Performance Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor training progress
tail -f checkpoints/detection_training/*/train.log
```

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [Weights & Biases](https://wandb.ai/)
- [Albumentations](https://albumentations.ai/)

## 🤝 Contributing

To contribute improvements to the training system:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This training system is part of the larger project and follows the same license terms.
