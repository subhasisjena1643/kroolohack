"""
Model Integration System
Integrates trained detection models with the existing attendance system
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import shutil
from datetime import datetime

from ultralytics import YOLO
from src.utils.logger import logger

class ModelIntegrator:
    """Integrates trained models with the existing system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path("models/detection")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.backup_dir = Path("models/backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry_file = self.models_dir / "model_registry.json"
        self.model_registry = self._load_model_registry()
        
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        try:
            if self.model_registry_file.exists():
                with open(self.model_registry_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "models": {},
                    "active_model": None,
                    "last_updated": None
                }
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            return {"models": {}, "active_model": None, "last_updated": None}
    
    def _save_model_registry(self):
        """Save model registry"""
        try:
            with open(self.model_registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def register_trained_model(self, 
                             model_path: str, 
                             model_name: str,
                             training_config: Dict[str, Any],
                             performance_metrics: Dict[str, float],
                             description: str = "") -> bool:
        """Register a newly trained model"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Generate unique model ID
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Copy model to models directory
            model_destination = self.models_dir / f"{model_id}.pt"
            shutil.copy2(model_path, model_destination)
            
            # Register model
            model_info = {
                "model_id": model_id,
                "model_name": model_name,
                "model_path": str(model_destination),
                "description": description,
                "training_config": training_config,
                "performance_metrics": performance_metrics,
                "registration_date": datetime.now().isoformat(),
                "model_size": model_destination.stat().st_size,
                "status": "registered"
            }
            
            self.model_registry["models"][model_id] = model_info
            self.model_registry["last_updated"] = datetime.now().isoformat()
            self._save_model_registry()
            
            logger.info(f"✅ Model registered: {model_id}")
            logger.info(f"   Performance: mAP50={performance_metrics.get('mAP50', 0):.4f}")
            logger.info(f"   Model size: {model_destination.stat().st_size / (1024*1024):.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Validate a registered model"""
        try:
            if model_id not in self.model_registry["models"]:
                return {"valid": False, "error": "Model not found in registry"}
            
            model_info = self.model_registry["models"][model_id]
            model_path = Path(model_info["model_path"])
            
            if not model_path.exists():
                return {"valid": False, "error": "Model file not found"}
            
            # Load and test model
            try:
                model = YOLO(str(model_path))
                
                # Test with dummy input
                dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                results = model(dummy_input, verbose=False)
                
                validation_result = {
                    "valid": True,
                    "model_loaded": True,
                    "inference_test": True,
                    "model_classes": len(model.names),
                    "expected_classes": 1,  # Person only
                    "model_size_mb": model_path.stat().st_size / (1024*1024),
                    "validation_date": datetime.now().isoformat()
                }
                
                # Check if model has correct number of classes
                if len(model.names) != 1 or 'person' not in model.names.values():
                    validation_result["warning"] = "Model may not be optimized for person detection"
                
                logger.info(f"✅ Model validation passed: {model_id}")
                return validation_result
                
            except Exception as model_error:
                return {
                    "valid": False,
                    "error": f"Model loading/inference failed: {model_error}"
                }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {"valid": False, "error": str(e)}
    
    def deploy_model(self, model_id: str, backup_current: bool = True) -> bool:
        """Deploy a model as the active detection model"""
        try:
            # Validate model first
            validation = self.validate_model(model_id)
            if not validation["valid"]:
                logger.error(f"Model validation failed: {validation['error']}")
                return False
            
            model_info = self.model_registry["models"][model_id]
            model_path = Path(model_info["model_path"])
            
            # Backup current active model if requested
            if backup_current and self.model_registry["active_model"]:
                self._backup_current_model()
            
            # Deploy new model
            active_model_path = self.models_dir / "active_detection_model.pt"
            shutil.copy2(model_path, active_model_path)
            
            # Update registry
            self.model_registry["active_model"] = model_id
            self.model_registry["last_updated"] = datetime.now().isoformat()
            self.model_registry["models"][model_id]["status"] = "active"
            self.model_registry["models"][model_id]["deployment_date"] = datetime.now().isoformat()
            
            # Mark other models as inactive
            for mid, minfo in self.model_registry["models"].items():
                if mid != model_id and minfo["status"] == "active":
                    minfo["status"] = "registered"
            
            self._save_model_registry()
            
            logger.info(f"🚀 Model deployed successfully: {model_id}")
            logger.info(f"   Active model path: {active_model_path}")
            logger.info(f"   Performance: mAP50={model_info['performance_metrics'].get('mAP50', 0):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False
    
    def _backup_current_model(self):
        """Backup the current active model"""
        try:
            active_model_path = self.models_dir / "active_detection_model.pt"
            if active_model_path.exists():
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                backup_path = self.backup_dir / backup_name
                shutil.copy2(active_model_path, backup_path)
                logger.info(f"📦 Current model backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"Error backing up current model: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        try:
            models = []
            for model_id, model_info in self.model_registry["models"].items():
                model_summary = {
                    "model_id": model_id,
                    "model_name": model_info["model_name"],
                    "status": model_info["status"],
                    "registration_date": model_info["registration_date"],
                    "performance": model_info["performance_metrics"],
                    "size_mb": model_info["model_size"] / (1024*1024)
                }
                models.append(model_summary)
            
            # Sort by registration date (newest first)
            models.sort(key=lambda x: x["registration_date"], reverse=True)
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the active model"""
        try:
            active_model_id = self.model_registry.get("active_model")
            if not active_model_id:
                return None
            
            if active_model_id in self.model_registry["models"]:
                return self.model_registry["models"][active_model_id]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting active model info: {e}")
            return None
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        try:
            comparison = {
                "models": {},
                "best_model": None,
                "comparison_date": datetime.now().isoformat()
            }
            
            best_map50 = 0
            best_model_id = None
            
            for model_id in model_ids:
                if model_id in self.model_registry["models"]:
                    model_info = self.model_registry["models"][model_id]
                    metrics = model_info["performance_metrics"]
                    
                    comparison["models"][model_id] = {
                        "model_name": model_info["model_name"],
                        "mAP50": metrics.get("mAP50", 0),
                        "mAP50_95": metrics.get("mAP50_95", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1_score": metrics.get("f1_score", 0),
                        "model_size_mb": model_info["model_size"] / (1024*1024)
                    }
                    
                    if metrics.get("mAP50", 0) > best_map50:
                        best_map50 = metrics.get("mAP50", 0)
                        best_model_id = model_id
            
            comparison["best_model"] = best_model_id
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def update_face_detector_config(self, model_id: str) -> bool:
        """Update face detector configuration to use new model"""
        try:
            # This would update the face detector configuration
            # to use the newly deployed model
            
            model_info = self.model_registry["models"][model_id]
            active_model_path = self.models_dir / "active_detection_model.pt"
            
            # Update configuration file or environment variable
            config_updates = {
                "model_path": str(active_model_path),
                "model_id": model_id,
                "model_performance": model_info["performance_metrics"],
                "last_updated": datetime.now().isoformat()
            }
            
            # Save configuration
            config_file = Path("configs/active_model_config.json")
            with open(config_file, 'w') as f:
                json.dump(config_updates, f, indent=2)
            
            logger.info(f"✅ Face detector configuration updated for model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating face detector config: {e}")
            return False
