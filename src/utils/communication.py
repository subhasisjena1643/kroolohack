"""
Communication Layer for Real-time Data Transmission
Handles WebSocket communication with Node.js backend
"""

import json
import time
import threading
from typing import Dict, Any, Optional, Callable
import websocket
import requests
from queue import Queue, Empty

from src.utils.logger import logger

class WebSocketClient:
    """WebSocket client for real-time communication with backend"""
    
    def __init__(self, url: str, reconnect_attempts: int = 5, reconnect_delay: float = 2.0):
        self.url = url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        self.is_running = False
        
        # Threading
        self.send_queue = Queue(maxsize=1000)
        self.send_thread = None
        
        # Callbacks
        self.on_message_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_attempts = 0
        self.last_error = None
    
    def connect(self) -> bool:
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to WebSocket server: {self.url}")
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start connection in separate thread
            self.is_running = True
            self.connection_thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            self.connection_thread.start()
            
            # Wait for connection
            timeout = 10.0
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                # Start send thread
                self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
                self.send_thread.start()
                
                logger.info("WebSocket connection established")
                return True
            else:
                logger.error("Failed to establish WebSocket connection within timeout")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.last_error = str(e)
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket server"""
        self.is_running = False
        self.is_connected = False
        
        if self.ws:
            self.ws.close()
        
        logger.info("WebSocket disconnected")
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to server"""
        if not self.is_connected:
            logger.warning("Cannot send message - not connected")
            return False
        
        try:
            self.send_queue.put_nowait(message)
            return True
        except:
            logger.warning("Send queue is full, dropping message")
            return False
    
    def _send_loop(self):
        """Background thread for sending messages"""
        while self.is_running:
            try:
                message = self.send_queue.get(timeout=1.0)
                
                if self.is_connected and self.ws:
                    json_message = json.dumps(message)
                    self.ws.send(json_message)
                    self.messages_sent += 1
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending message: {e}")
    
    def _on_open(self, ws):
        """WebSocket open callback"""
        self.is_connected = True
        self.connection_attempts += 1
        logger.info("WebSocket connection opened")
        
        if self.on_connect_callback:
            try:
                self.on_connect_callback()
            except Exception as e:
                logger.error(f"Error in connect callback: {e}")
    
    def _on_message(self, ws, message):
        """WebSocket message callback"""
        try:
            self.messages_received += 1
            data = json.loads(message)
            
            if self.on_message_callback:
                self.on_message_callback(data)
                
        except Exception as e:
            logger.error(f"Error processing received message: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error callback"""
        logger.error(f"WebSocket error: {error}")
        self.last_error = str(error)
        
        if self.on_error_callback:
            try:
                self.on_error_callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket close callback"""
        self.is_connected = False
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        
        if self.on_disconnect_callback:
            try:
                self.on_disconnect_callback(close_status_code, close_msg)
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
        
        # Attempt reconnection if still running
        if self.is_running and self.connection_attempts < self.reconnect_attempts:
            logger.info(f"Attempting reconnection in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
            self.connect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'is_connected': self.is_connected,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'connection_attempts': self.connection_attempts,
            'queue_size': self.send_queue.qsize(),
            'last_error': self.last_error
        }

class APIClient:
    """REST API client for backend communication"""
    
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Statistics
        self.requests_sent = 0
        self.requests_failed = 0
        self.last_error = None
    
    def send_engagement_data(self, data: Dict[str, Any]) -> bool:
        """Send engagement data to API"""
        try:
            url = f"{self.base_url}/engagement"
            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            self.requests_sent += 1
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"API request failed with status {response.status_code}")
                self.requests_failed += 1
                return False
                
        except Exception as e:
            logger.error(f"Error sending API request: {e}")
            self.last_error = str(e)
            self.requests_failed += 1
            return False
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration from API"""
        try:
            url = f"{self.base_url}/config"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Config request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            self.last_error = str(e)
            return None
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=self.timeout)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API client statistics"""
        return {
            'requests_sent': self.requests_sent,
            'requests_failed': self.requests_failed,
            'success_rate': (self.requests_sent - self.requests_failed) / self.requests_sent if self.requests_sent > 0 else 0.0,
            'last_error': self.last_error
        }

class CommunicationManager:
    """Manages all communication with backend"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize clients
        self.websocket_client = WebSocketClient(
            url=config.get('websocket_url', 'ws://localhost:3000/engagement'),
            reconnect_attempts=config.get('reconnect_attempts', 5),
            reconnect_delay=config.get('reconnect_delay', 2.0)
        )

        self.api_client = APIClient(
            base_url=config.get('api_base_url', 'http://localhost:3000/api'),
            timeout=config.get('api_timeout', 5.0)
        )

        # Data formatting
        self.send_interval = config.get('send_interval', 1.0)
        self.batch_size = config.get('batch_size', 10)

        # Data queue and sending
        self.data_queue = Queue(maxsize=1000)
        self.send_thread = None
        self.is_running = False

        # Setup callbacks
        self.websocket_client.on_connect_callback = self._on_websocket_connect
        self.websocket_client.on_disconnect_callback = self._on_websocket_disconnect
        self.websocket_client.on_error_callback = self._on_websocket_error
    
    def start(self) -> bool:
        """Start communication manager"""
        logger.info("Starting communication manager...")
        
        # Check API health
        if not self.api_client.health_check():
            logger.warning("Backend API health check failed")
        
        # Connect WebSocket
        if not self.websocket_client.connect():
            logger.error("Failed to connect WebSocket")
            return False
        
        # Start data sending thread
        self.is_running = True
        self.send_thread = threading.Thread(target=self._data_send_loop, daemon=True)
        self.send_thread.start()
        
        logger.info("Communication manager started")
        return True
    
    def stop(self):
        """Stop communication manager"""
        self.is_running = False
        
        if self.websocket_client:
            self.websocket_client.disconnect()
        
        logger.info("Communication manager stopped")
    
    def send_engagement_data(self, data: Dict[str, Any]):
        """Send engagement data"""
        try:
            # Add timestamp
            data['timestamp'] = time.time()
            
            # Add to queue
            self.data_queue.put_nowait(data)
            
        except:
            logger.warning("Data queue is full, dropping engagement data")
    
    def _data_send_loop(self):
        """Background thread for sending data"""
        while self.is_running:
            try:
                # Collect data for batch sending
                batch_data = []
                
                # Get data from queue
                try:
                    data = self.data_queue.get(timeout=self.send_interval)
                    batch_data.append(data)
                    
                    # Collect more data for batch (non-blocking)
                    while len(batch_data) < self.batch_size:
                        try:
                            additional_data = self.data_queue.get_nowait()
                            batch_data.append(additional_data)
                        except Empty:
                            break
                            
                except Empty:
                    continue
                
                # Send data
                if batch_data:
                    self._send_batch_data(batch_data)
                
            except Exception as e:
                logger.error(f"Error in data send loop: {e}")
                time.sleep(1.0)
    
    def _send_batch_data(self, batch_data: List[Dict[str, Any]]):
        """Send batch of data"""
        try:
            # Format data for transmission
            formatted_data = {
                'type': 'engagement_batch',
                'timestamp': time.time(),
                'data_count': len(batch_data),
                'data': batch_data
            }
            
            # Send via WebSocket (primary)
            if self.websocket_client.is_connected:
                self.websocket_client.send_message(formatted_data)
            else:
                # Fallback to API
                for data in batch_data:
                    self.api_client.send_engagement_data(data)
                    
        except Exception as e:
            logger.error(f"Error sending batch data: {e}")
    
    def _on_websocket_connect(self):
        """WebSocket connect callback"""
        logger.info("WebSocket connected - sending initial handshake")
        
        handshake_data = {
            'type': 'handshake',
            'client_type': 'engagement_analyzer',
            'timestamp': time.time(),
            'version': '1.0.0'
        }
        
        self.websocket_client.send_message(handshake_data)
    
    def _on_websocket_disconnect(self, code, message):
        """WebSocket disconnect callback"""
        logger.warning(f"WebSocket disconnected: {code} - {message}")
    
    def _on_websocket_error(self, error):
        """WebSocket error callback"""
        logger.error(f"WebSocket error: {error}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication status"""
        return {
            'websocket': self.websocket_client.get_stats(),
            'api': self.api_client.get_stats(),
            'data_queue_size': self.data_queue.qsize(),
            'is_running': self.is_running
        }
