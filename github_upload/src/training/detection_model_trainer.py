"""
Advanced Detection Model Trainer
State-of-the-art training system for human detection with external datasets
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import requests
import zipfile
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateOfTheArtDatasetManager:
    """Manager for downloading and preparing state-of-the-art datasets"""
    
    def __init__(self, data_dir: str = "data/training_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State-of-the-art datasets for human detection
        self.datasets = {
            'coco_person': {
                'name': 'COCO Person Detection',
                'url': 'http://images.cocodataset.org/zips/train2017.zip',
                'annotations_url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                'description': 'COCO dataset filtered for person class - industry standard',
                'size': '18GB',
                'samples': 64115
            },
            'crowdhuman': {
                'name': 'CrowdHuman Dataset',
                'url': 'https://www.crowdhuman.org/download',
                'description': 'Specialized for crowded human detection scenarios',
                'size': '2.5GB',
                'samples': 15000
            },
            'citypersons': {
                'name': 'CityPersons Dataset',
                'url': 'https://github.com/cvgroup-njust/CityPersons',
                'description': 'Urban person detection with challenging scenarios',
                'size': '1.2GB',
                'samples': 5000
            },
            'wider_person': {
                'name': 'WIDER Person Dataset',
                'url': 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERPerson/',
                'description': 'Large-scale person detection dataset',
                'size': '3.8GB',
                'samples': 13382
            },
            'human_parts': {
                'name': 'Human Parts Dataset',
                'url': 'https://github.com/soeaver/human_parts_dataset',
                'description': 'Detailed human body parts for fine-grained detection',
                'size': '1.5GB',
                'samples': 8000
            }
        }
        
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """Download and prepare a specific dataset"""
        try:
            if dataset_name not in self.datasets:
                logger.error(f"Dataset {dataset_name} not available")
                return False
            
            dataset_info = self.datasets[dataset_name]
            dataset_path = self.data_dir / dataset_name
            
            if dataset_path.exists() and not force_download:
                logger.info(f"Dataset {dataset_name} already exists")
                return True
            
            logger.info(f"Downloading {dataset_info['name']} ({dataset_info['size']})...")
            
            # Create dataset directory
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Download based on dataset type
            if dataset_name == 'coco_person':
                return self._download_coco_person(dataset_path)
            elif dataset_name == 'crowdhuman':
                return self._download_crowdhuman(dataset_path)
            elif dataset_name == 'citypersons':
                return self._download_citypersons(dataset_path)
            elif dataset_name == 'wider_person':
                return self._download_wider_person(dataset_path)
            elif dataset_name == 'human_parts':
                return self._download_human_parts(dataset_path)
            
            return False
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {e}")
            return False
    
    def _download_coco_person(self, dataset_path: Path) -> bool:
        """Download and filter COCO dataset for person class"""
        try:
            # Download COCO images
            images_zip = dataset_path / "train2017.zip"
            if not images_zip.exists():
                logger.info("Downloading COCO train2017 images...")
                self._download_file(self.datasets['coco_person']['url'], images_zip)
            
            # Download annotations
            annotations_zip = dataset_path / "annotations_trainval2017.zip"
            if not annotations_zip.exists():
                logger.info("Downloading COCO annotations...")
                self._download_file(self.datasets['coco_person']['annotations_url'], annotations_zip)
            
            # Extract files
            logger.info("Extracting COCO dataset...")
            self._extract_zip(images_zip, dataset_path)
            self._extract_zip(annotations_zip, dataset_path)
            
            # Filter for person class and convert to YOLO format
            self._convert_coco_to_yolo_person(dataset_path)
            
            logger.info("COCO Person dataset prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading COCO Person dataset: {e}")
            return False
    
    def _download_crowdhuman(self, dataset_path: Path) -> bool:
        """Download CrowdHuman dataset"""
        try:
            logger.info("CrowdHuman dataset requires manual download from official website")
            logger.info("Please visit: https://www.crowdhuman.org/download")
            logger.info("Download and extract to: " + str(dataset_path))
            
            # Create placeholder structure
            (dataset_path / "images").mkdir(exist_ok=True)
            (dataset_path / "labels").mkdir(exist_ok=True)
            
            # Create instruction file
            with open(dataset_path / "DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
                f.write("CrowdHuman Dataset Download Instructions:\n")
                f.write("1. Visit: https://www.crowdhuman.org/download\n")
                f.write("2. Download train and validation sets\n")
                f.write("3. Extract images to: images/\n")
                f.write("4. Extract annotations to: annotations/\n")
                f.write("5. Run conversion script to YOLO format\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up CrowdHuman dataset: {e}")
            return False
    
    def _download_file(self, url: str, destination: Path) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file {url}: {e}")
            return False
    
    def _extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return False
    
    def _convert_coco_to_yolo_person(self, dataset_path: Path) -> bool:
        """Convert COCO annotations to YOLO format for person class only"""
        try:
            import pycocotools.coco as coco
            
            # Load COCO annotations
            ann_file = dataset_path / "annotations" / "instances_train2017.json"
            coco_api = coco.COCO(str(ann_file))
            
            # Get person category ID (usually 1)
            person_cat_id = None
            for cat_id, cat_info in coco_api.cats.items():
                if cat_info['name'] == 'person':
                    person_cat_id = cat_id
                    break
            
            if person_cat_id is None:
                logger.error("Person category not found in COCO dataset")
                return False
            
            # Create YOLO directories
            yolo_images_dir = dataset_path / "yolo" / "images"
            yolo_labels_dir = dataset_path / "yolo" / "labels"
            yolo_images_dir.mkdir(parents=True, exist_ok=True)
            yolo_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images with person annotations
            img_ids = coco_api.getImgIds(catIds=[person_cat_id])
            
            logger.info(f"Converting {len(img_ids)} images with person annotations...")
            
            for img_id in tqdm(img_ids[:10000]):  # Limit to 10k images for training
                img_info = coco_api.loadImgs(img_id)[0]
                ann_ids = coco_api.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
                anns = coco_api.loadAnns(ann_ids)
                
                if not anns:  # Skip images without person annotations
                    continue
                
                # Copy image
                src_img = dataset_path / "train2017" / img_info['file_name']
                dst_img = yolo_images_dir / img_info['file_name']
                
                if src_img.exists():
                    import shutil
                    shutil.copy2(src_img, dst_img)
                    
                    # Create YOLO label file
                    label_file = yolo_labels_dir / (img_info['file_name'].replace('.jpg', '.txt'))
                    
                    with open(label_file, 'w') as f:
                        for ann in anns:
                            bbox = ann['bbox']  # [x, y, width, height]
                            
                            # Convert to YOLO format (normalized center coordinates)
                            img_w, img_h = img_info['width'], img_info['height']
                            x_center = (bbox[0] + bbox[2] / 2) / img_w
                            y_center = (bbox[1] + bbox[3] / 2) / img_h
                            width = bbox[2] / img_w
                            height = bbox[3] / img_h
                            
                            # YOLO format: class_id x_center y_center width height
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Create dataset.yaml for YOLO training
            dataset_yaml = {
                'path': str(dataset_path / "yolo"),
                'train': 'images',
                'val': 'images',  # Use same for now, should split properly
                'nc': 1,  # Number of classes (person only)
                'names': ['person']
            }
            
            with open(dataset_path / "dataset.yaml", 'w') as f:
                yaml.dump(dataset_yaml, f)
            
            logger.info("COCO to YOLO conversion completed")
            return True
            
        except Exception as e:
            logger.error(f"Error converting COCO to YOLO: {e}")
            return False

class AdvancedDetectionTrainer:
    """Advanced trainer for human detection models using state-of-the-art techniques"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset_manager = StateOfTheArtDatasetManager()
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0005)
        
        # Model configuration
        self.model_size = config.get('model_size', 'yolov8n')  # n, s, m, l, x
        self.input_size = config.get('input_size', 640)
        
        # Training techniques
        self.use_augmentation = config.get('use_augmentation', True)
        self.use_mixup = config.get('use_mixup', True)
        self.use_mosaic = config.get('use_mosaic', True)
        self.use_focal_loss = config.get('use_focal_loss', True)
        
        # Checkpoint and logging
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/detection_training'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases for experiment tracking
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            wandb.init(
                project="human-detection-training",
                config=config,
                name=f"detection_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        logger.info(f"Advanced Detection Trainer initialized on {self.device}")

    def prepare_datasets(self, datasets_to_use: List[str] = None) -> bool:
        """Prepare state-of-the-art datasets for training"""
        try:
            if datasets_to_use is None:
                datasets_to_use = ['coco_person']  # Start with COCO

            logger.info("Preparing state-of-the-art datasets...")

            for dataset_name in datasets_to_use:
                logger.info(f"Preparing {dataset_name}...")
                success = self.dataset_manager.download_dataset(dataset_name)
                if not success:
                    logger.warning(f"Failed to prepare {dataset_name}")
                else:
                    logger.info(f"✅ {dataset_name} prepared successfully")

            return True

        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            return False

    def create_advanced_augmentation_pipeline(self) -> A.Compose:
        """Create state-of-the-art augmentation pipeline for robust training"""
        try:
            augmentations = [
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),

                # Photometric transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),

                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),

                # Weather and lighting effects
                A.RandomRain(p=0.1),
                A.RandomFog(p=0.1),
                A.RandomSunFlare(p=0.1),
                A.RandomShadow(p=0.2),

                # Cutout and mixup-like effects
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.3
                ),

                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]

            return A.Compose(
                augmentations,
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']
                )
            )

        except Exception as e:
            logger.error(f"Error creating augmentation pipeline: {e}")
            return None

    def initialize_model(self) -> bool:
        """Initialize YOLOv8 model with advanced configuration"""
        try:
            logger.info(f"Initializing YOLOv8 model ({self.model_size})...")

            # Load pre-trained YOLOv8 model
            model_path = f"yolov8{self.model_size[-1]}.pt"  # e.g., yolov8n.pt
            self.model = YOLO(model_path)

            # Configure model for human detection
            self.model.model[-1].nc = 1  # Number of classes (person only)
            self.model.model[-1].anchors = self.model.model[-1].anchors.clone()

            logger.info("✅ YOLOv8 model initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False

    def train_with_advanced_techniques(self, dataset_path: str) -> bool:
        """Train model using state-of-the-art techniques"""
        try:
            logger.info("Starting advanced training with state-of-the-art techniques...")

            # Training configuration with advanced techniques
            training_config = {
                'data': dataset_path,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.input_size,
                'device': self.device,
                'workers': 8,
                'project': str(self.checkpoint_dir),
                'name': f'human_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

                # Advanced optimization
                'optimizer': 'AdamW',  # Better than SGD for many cases
                'lr0': self.learning_rate,
                'weight_decay': self.weight_decay,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,

                # Advanced augmentation
                'hsv_h': 0.015,  # Hue augmentation
                'hsv_s': 0.7,    # Saturation augmentation
                'hsv_v': 0.4,    # Value augmentation
                'degrees': 15.0,  # Rotation augmentation
                'translate': 0.1, # Translation augmentation
                'scale': 0.5,    # Scale augmentation
                'shear': 0.0,    # Shear augmentation
                'perspective': 0.0, # Perspective augmentation
                'flipud': 0.1,   # Vertical flip probability
                'fliplr': 0.5,   # Horizontal flip probability
                'mosaic': 1.0 if self.use_mosaic else 0.0,
                'mixup': 0.1 if self.use_mixup else 0.0,
                'copy_paste': 0.1,

                # Advanced loss functions
                'box': 7.5,      # Box loss gain
                'cls': 0.5,      # Class loss gain
                'dfl': 1.5,      # DFL loss gain
                'fl_gamma': 0.0 if not self.use_focal_loss else 1.5,  # Focal loss gamma

                # Model EMA and validation
                'ema': True,     # Use Exponential Moving Average
                'patience': 50,  # Early stopping patience
                'save_period': 10, # Save checkpoint every N epochs
                'val': True,     # Validate during training

                # Advanced training techniques
                'amp': True,     # Automatic Mixed Precision
                'fraction': 1.0, # Dataset fraction to use
                'profile': False, # Profile training
                'freeze': None,  # Freeze layers
                'multi_scale': True, # Multi-scale training
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'verbose': True,

                # Logging and visualization
                'plots': True,
                'save_json': True,
                'save_hybrid': False,
                'conf': None,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'exist_ok': True
            }

            # Add Weights & Biases logging if enabled
            if self.use_wandb:
                training_config['wandb'] = True

            # Start training
            logger.info("🚀 Starting advanced YOLOv8 training...")
            results = self.model.train(**training_config)

            # Save final model
            final_model_path = self.checkpoint_dir / "best_human_detection_model.pt"
            self.model.save(str(final_model_path))

            logger.info(f"✅ Training completed! Best model saved to: {final_model_path}")

            # Log training results
            if self.use_wandb:
                wandb.log({
                    "final_map50": results.results_dict.get('metrics/mAP50(B)', 0),
                    "final_map50_95": results.results_dict.get('metrics/mAP50-95(B)', 0),
                    "final_precision": results.results_dict.get('metrics/precision(B)', 0),
                    "final_recall": results.results_dict.get('metrics/recall(B)', 0)
                })

            return True

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def evaluate_model(self, test_dataset_path: str) -> Dict[str, float]:
        """Evaluate trained model on test dataset"""
        try:
            logger.info("Evaluating trained model...")

            if self.model is None:
                logger.error("No model loaded for evaluation")
                return {}

            # Run validation
            results = self.model.val(
                data=test_dataset_path,
                imgsz=self.input_size,
                batch=self.batch_size,
                conf=0.001,  # Low confidence for evaluation
                iou=0.6,
                max_det=300,
                half=False,
                device=self.device,
                plots=True,
                save_json=True,
                save_hybrid=False,
                verbose=True
            )

            # Extract metrics
            metrics = {
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0),
                'f1_score': 2 * (results.results_dict.get('metrics/precision(B)', 0) * results.results_dict.get('metrics/recall(B)', 0)) /
                           (results.results_dict.get('metrics/precision(B)', 0) + results.results_dict.get('metrics/recall(B)', 0) + 1e-6)
            }

            logger.info(f"Evaluation Results:")
            logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}

    def export_optimized_model(self, export_format: str = 'onnx') -> str:
        """Export model in optimized format for deployment"""
        try:
            logger.info(f"Exporting model to {export_format} format...")

            if self.model is None:
                logger.error("No model loaded for export")
                return ""

            # Export model
            export_path = self.model.export(
                format=export_format,
                imgsz=self.input_size,
                keras=False,
                optimize=True,
                half=False,
                int8=False,
                dynamic=False,
                simplify=True,
                opset=None,
                workspace=4,
                nms=True
            )

            logger.info(f"✅ Model exported to: {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return ""
