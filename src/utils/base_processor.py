"""
Base processor class for all AI modules
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from queue import Queue, Empty
import numpy as np

from utils.logger import logger

class BaseProcessor(ABC):
    """Base class for all processing modules"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_running = False
        self.is_initialized = False
        
        # Threading
        self.process_thread: Optional[threading.Thread] = None
        self.input_queue: Queue = Queue(maxsize=100)
        self.output_queue: Queue = Queue(maxsize=100)
        
        # Performance tracking
        self.process_count = 0
        self.total_process_time = 0.0
        self.last_process_time = 0.0
        
        # Callbacks
        self.result_callback: Optional[Callable] = None
        
        logger.info(f"Initialized {self.name} processor")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the processor (load models, setup resources)"""
        pass
    
    @abstractmethod
    def process_data(self, data: Any) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def start(self) -> bool:
        """Start the processor"""
        if self.is_running:
            logger.warning(f"{self.name} processor is already running")
            return True
        
        if not self.is_initialized:
            if not self.initialize():
                logger.error(f"Failed to initialize {self.name} processor")
                return False
            self.is_initialized = True
        
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info(f"Started {self.name} processor")
        return True
    
    def stop(self):
        """Stop the processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
        
        self.cleanup()
        logger.info(f"Stopped {self.name} processor")
    
    def add_data(self, data: Any) -> bool:
        """Add data to processing queue"""
        try:
            self.input_queue.put_nowait(data)
            return True
        except:
            logger.warning(f"{self.name} input queue is full, dropping data")
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get processed result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def set_result_callback(self, callback: Callable):
        """Set callback function for results"""
        self.result_callback = callback
    
    def _process_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get data from input queue
                data = self.input_queue.get(timeout=0.1)
                
                # Process data
                start_time = time.time()
                result = self.process_data(data)
                process_time = time.time() - start_time
                
                # Update performance metrics
                self._update_performance_metrics(process_time)
                
                # Add timestamp and performance info
                result.update({
                    'timestamp': time.time(),
                    'processor': self.name,
                    'process_time': process_time
                })
                
                # Send result to output queue
                try:
                    self.output_queue.put_nowait(result)
                except:
                    logger.warning(f"{self.name} output queue is full, dropping result")
                
                # Call result callback if set
                if self.result_callback:
                    try:
                        self.result_callback(result)
                    except Exception as e:
                        logger.error(f"Error in result callback for {self.name}: {e}")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in {self.name} processing loop: {e}")
    
    def _update_performance_metrics(self, process_time: float):
        """Update performance tracking metrics"""
        self.process_count += 1
        self.total_process_time += process_time
        self.last_process_time = process_time
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.process_count == 0:
            return {
                'avg_process_time': 0.0,
                'last_process_time': 0.0,
                'total_processed': 0,
                'fps': 0.0
            }
        
        avg_time = self.total_process_time / self.process_count
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_process_time': avg_time,
            'last_process_time': self.last_process_time,
            'total_processed': self.process_count,
            'fps': fps
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.process_count = 0
        self.total_process_time = 0.0
        self.last_process_time = 0.0
