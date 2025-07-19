"""
Advanced Model Checkpoint Management System
Handles model saving, loading, and continuous training progression
Implements state-of-the-art checkpoint management with versioning
"""

import os
import pickle
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime
import hashlib

from src.utils.logger import logger

class ModelCheckpointManager:
    """Advanced checkpoint management for continuous learning"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint structure
        self.models_dir = self.checkpoint_dir / "models"
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.training_data_dir = self.checkpoint_dir / "training_data"
        self.performance_dir = self.checkpoint_dir / "performance"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.training_data_dir, self.performance_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Checkpoint metadata
        self.checkpoint_metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_training_samples': 0,
            'training_epochs': 0,
            'model_versions': {},
            'performance_history': [],
            'best_performance': {}
        }
        
        # Load existing metadata if available
        self._load_checkpoint_metadata()
    
    def save_model_checkpoint(self, model_name: str, model_object: Any, 
                            training_data: Dict[str, Any], 
                            performance_metrics: Dict[str, float],
                            model_config: Dict[str, Any] = None) -> str:
        """Save complete model checkpoint with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"{model_name}_{timestamp}"
            
            # Create checkpoint directory
            checkpoint_path = self.models_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model object
            model_file = checkpoint_path / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_object, f)
            
            # Save training data
            training_data_file = checkpoint_path / "training_data.json"
            with open(training_data_file, 'w') as f:
                json.dump(training_data, f, indent=2, default=str)
            
            # Save model configuration
            if model_config:
                config_file = checkpoint_path / "config.json"
                with open(config_file, 'w') as f:
                    json.dump(model_config, f, indent=2)
            
            # Save performance metrics
            performance_file = checkpoint_path / "performance.json"
            with open(performance_file, 'w') as f:
                json.dump(performance_metrics, f, indent=2)
            
            # Update metadata
            self._update_checkpoint_metadata(checkpoint_id, model_name, performance_metrics, training_data)
            
            # Create symlink to latest
            latest_link = self.models_dir / f"{model_name}_latest"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_path.name)
            
            logger.info(f"Model checkpoint saved: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {e}")
            return None
    
    def load_latest_checkpoint(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a model"""
        try:
            latest_link = self.models_dir / f"{model_name}_latest"
            if not latest_link.exists():
                logger.info(f"No checkpoint found for model: {model_name}")
                return None
            
            checkpoint_path = self.models_dir / latest_link.readlink()
            
            # Load model
            model_file = checkpoint_path / "model.pkl"
            if not model_file.exists():
                logger.error(f"Model file not found in checkpoint: {checkpoint_path}")
                return None
            
            with open(model_file, 'rb') as f:
                model_object = pickle.load(f)
            
            # Load training data
            training_data = {}
            training_data_file = checkpoint_path / "training_data.json"
            if training_data_file.exists():
                with open(training_data_file, 'r') as f:
                    training_data = json.load(f)
            
            # Load configuration
            config = {}
            config_file = checkpoint_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            # Load performance metrics
            performance = {}
            performance_file = checkpoint_path / "performance.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance = json.load(f)
            
            checkpoint_data = {
                'model': model_object,
                'training_data': training_data,
                'config': config,
                'performance': performance,
                'checkpoint_id': checkpoint_path.name,
                'model_name': model_name
            }
            
            logger.info(f"Loaded checkpoint for {model_name}: {checkpoint_path.name}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint for {model_name}: {e}")
            return None
    
    def get_checkpoint_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get checkpoint history for a model"""
        history = []
        
        for checkpoint_dir in self.models_dir.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith(model_name):
                performance_file = checkpoint_dir / "performance.json"
                if performance_file.exists():
                    with open(performance_file, 'r') as f:
                        performance = json.load(f)
                    
                    history.append({
                        'checkpoint_id': checkpoint_dir.name,
                        'timestamp': checkpoint_dir.stat().st_mtime,
                        'performance': performance
                    })
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history
    
    def cleanup_old_checkpoints(self, model_name: str, keep_count: int = 5):
        """Clean up old checkpoints, keeping only the most recent ones"""
        try:
            history = self.get_checkpoint_history(model_name)
            
            if len(history) <= keep_count:
                return
            
            # Remove old checkpoints
            for checkpoint in history[keep_count:]:
                checkpoint_path = self.models_dir / checkpoint['checkpoint_id']
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint['checkpoint_id']}")
            
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")
    
    def _load_checkpoint_metadata(self):
        """Load checkpoint metadata from file"""
        metadata_file = self.metadata_dir / "checkpoint_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.checkpoint_metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading checkpoint metadata: {e}")
    
    def _update_checkpoint_metadata(self, checkpoint_id: str, model_name: str, 
                                  performance_metrics: Dict[str, float], 
                                  training_data: Dict[str, Any]):
        """Update checkpoint metadata"""
        try:
            # Update metadata
            self.checkpoint_metadata['last_updated'] = datetime.now().isoformat()
            self.checkpoint_metadata['total_training_samples'] += len(training_data.get('samples', []))
            self.checkpoint_metadata['training_epochs'] += 1
            
            # Update model versions
            if model_name not in self.checkpoint_metadata['model_versions']:
                self.checkpoint_metadata['model_versions'][model_name] = []
            
            self.checkpoint_metadata['model_versions'][model_name].append({
                'checkpoint_id': checkpoint_id,
                'timestamp': datetime.now().isoformat(),
                'performance': performance_metrics,
                'training_samples': len(training_data.get('samples', []))
            })
            
            # Update performance history
            self.checkpoint_metadata['performance_history'].append({
                'model_name': model_name,
                'checkpoint_id': checkpoint_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': performance_metrics
            })
            
            # Update best performance
            for metric, value in performance_metrics.items():
                if metric not in self.checkpoint_metadata['best_performance']:
                    self.checkpoint_metadata['best_performance'][metric] = {
                        'value': value,
                        'model_name': model_name,
                        'checkpoint_id': checkpoint_id
                    }
                elif value > self.checkpoint_metadata['best_performance'][metric]['value']:
                    self.checkpoint_metadata['best_performance'][metric] = {
                        'value': value,
                        'model_name': model_name,
                        'checkpoint_id': checkpoint_id
                    }
            
            # Save metadata
            metadata_file = self.metadata_dir / "checkpoint_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.checkpoint_metadata, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error updating checkpoint metadata: {e}")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get overall training progress"""
        return {
            'total_training_samples': self.checkpoint_metadata['total_training_samples'],
            'training_epochs': self.checkpoint_metadata['training_epochs'],
            'model_count': len(self.checkpoint_metadata['model_versions']),
            'best_performance': self.checkpoint_metadata['best_performance'],
            'last_updated': self.checkpoint_metadata['last_updated']
        }
    
    def export_checkpoint(self, checkpoint_id: str, export_path: str) -> bool:
        """Export a checkpoint for sharing or deployment"""
        try:
            checkpoint_path = self.models_dir / checkpoint_id
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            # Create export archive
            shutil.make_archive(export_path, 'zip', checkpoint_path)
            logger.info(f"Checkpoint exported to: {export_path}.zip")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting checkpoint: {e}")
            return False
