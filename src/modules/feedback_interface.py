"""
Feedback Interface for Continuous Learning
Provides web interface and API for teacher/user feedback collection
Enables real-time model improvement through human feedback
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import Flask, render_template, request, jsonify
import threading
from collections import deque
import sqlite3
from pathlib import Path

from src.utils.logger import logger

@dataclass
class FeedbackEntry:
    """Single feedback entry from user"""
    timestamp: float
    session_id: str
    student_id: Optional[str]
    predicted_engagement: str
    actual_engagement: str
    confidence: float
    feedback_type: str
    comments: Optional[str]
    teacher_id: Optional[str]
    context: Dict[str, Any]

class FeedbackInterface:
    """Web interface for collecting teacher/user feedback"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port = config.get('feedback_port', 5001)
        self.host = config.get('feedback_host', '127.0.0.1')
        
        # Data storage
        self.data_dir = Path("data")
        self.feedback_db_path = self.data_dir / "feedback.db"
        
        # Create data directory
        self.data_dir.mkdir(exist_ok=True)
        
        # Feedback storage
        self.feedback_queue = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=5000)
        
        # Flask app
        self.app = Flask(__name__, template_folder='templates')
        self._setup_routes()
        
        # Database
        self._init_database()
        
        # Server thread
        self.server_thread = None
        self.running = False
        
        # Callbacks for feedback processing
        self.feedback_callbacks = []
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    session_id TEXT,
                    student_id TEXT,
                    predicted_engagement TEXT,
                    actual_engagement TEXT,
                    confidence REAL,
                    feedback_type TEXT,
                    comments TEXT,
                    teacher_id TEXT,
                    context TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Feedback database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing feedback database: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes for feedback interface"""
        
        @self.app.route('/')
        def index():
            """Main feedback interface"""
            return render_template('feedback_interface.html')
        
        @self.app.route('/api/current_predictions')
        def get_current_predictions():
            """Get current engagement predictions for feedback"""
            try:
                # This would be populated by the main system
                predictions = self._get_latest_predictions()
                return jsonify(predictions)
            except Exception as e:
                logger.error(f"Error getting current predictions: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/submit_feedback', methods=['POST'])
        def submit_feedback():
            """Submit feedback for a prediction"""
            try:
                data = request.json
                
                feedback_entry = FeedbackEntry(
                    timestamp=time.time(),
                    session_id=data.get('session_id', 'unknown'),
                    student_id=data.get('student_id'),
                    predicted_engagement=data.get('predicted_engagement', ''),
                    actual_engagement=data.get('actual_engagement', ''),
                    confidence=float(data.get('confidence', 0.0)),
                    feedback_type=data.get('feedback_type', 'teacher'),
                    comments=data.get('comments'),
                    teacher_id=data.get('teacher_id'),
                    context=data.get('context', {})
                )
                
                # Store feedback
                self._store_feedback(feedback_entry)
                
                # Add to processing queue
                self.feedback_queue.append(feedback_entry)
                
                # Notify callbacks
                self._notify_feedback_callbacks(feedback_entry)
                
                return jsonify({'status': 'success', 'message': 'Feedback submitted successfully'})
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/feedback_stats')
        def get_feedback_stats():
            """Get feedback statistics"""
            try:
                stats = self._calculate_feedback_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting feedback stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/model_performance')
        def get_model_performance():
            """Get current model performance metrics"""
            try:
                performance = self._get_model_performance()
                return jsonify(performance)
            except Exception as e:
                logger.error(f"Error getting model performance: {e}")
                return jsonify({'error': str(e)}), 500
    
    def start_server(self):
        """Start feedback server in background thread"""
        if self.running:
            logger.warning("Feedback server already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        logger.info(f"Feedback interface started at http://{self.host}:{self.port}")
    
    def stop_server(self):
        """Stop feedback server"""
        self.running = False
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        logger.info("Feedback server stopped")
    
    def _run_server(self):
        """Run Flask server"""
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Error running feedback server: {e}")
    
    def add_feedback_callback(self, callback):
        """Add callback function for feedback processing"""
        self.feedback_callbacks.append(callback)
    
    def _notify_feedback_callbacks(self, feedback_entry: FeedbackEntry):
        """Notify all registered callbacks about new feedback"""
        for callback in self.feedback_callbacks:
            try:
                callback(feedback_entry)
            except Exception as e:
                logger.error(f"Error in feedback callback: {e}")
    
    def _store_feedback(self, feedback_entry: FeedbackEntry):
        """Store feedback in database"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (
                    timestamp, session_id, student_id, predicted_engagement,
                    actual_engagement, confidence, feedback_type, comments,
                    teacher_id, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_entry.timestamp,
                feedback_entry.session_id,
                feedback_entry.student_id,
                feedback_entry.predicted_engagement,
                feedback_entry.actual_engagement,
                feedback_entry.confidence,
                feedback_entry.feedback_type,
                feedback_entry.comments,
                feedback_entry.teacher_id,
                json.dumps(feedback_entry.context)
            ))
            
            conn.commit()
            conn.close()
            
            # Add to history
            self.feedback_history.append(feedback_entry)
            
            logger.info(f"Stored feedback: {feedback_entry.predicted_engagement} -> {feedback_entry.actual_engagement}")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    def _get_latest_predictions(self) -> List[Dict[str, Any]]:
        """Get latest engagement predictions for feedback interface"""
        # This would be populated by the main system
        # For now, return mock data
        return [
            {
                'student_id': 'student_1',
                'predicted_engagement': 'high_engagement',
                'confidence': 0.85,
                'timestamp': time.time(),
                'features': {
                    'head_stability': 0.8,
                    'eye_focus': 0.9,
                    'posture': 0.7
                }
            },
            {
                'student_id': 'student_2',
                'predicted_engagement': 'medium_engagement',
                'confidence': 0.65,
                'timestamp': time.time(),
                'features': {
                    'head_stability': 0.6,
                    'eye_focus': 0.7,
                    'posture': 0.5
                }
            }
        ]
    
    def _calculate_feedback_stats(self) -> Dict[str, Any]:
        """Calculate feedback statistics"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            # Total feedback count
            cursor.execute('SELECT COUNT(*) FROM feedback')
            total_feedback = cursor.fetchone()[0]
            
            # Feedback by type
            cursor.execute('SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type')
            feedback_by_type = dict(cursor.fetchall())
            
            # Accuracy calculation
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_engagement = actual_engagement THEN 1 ELSE 0 END) as correct
                FROM feedback 
                WHERE actual_engagement IS NOT NULL
            ''')
            accuracy_data = cursor.fetchone()
            accuracy = (accuracy_data[1] / accuracy_data[0]) if accuracy_data[0] > 0 else 0.0
            
            # Recent feedback (last 24 hours)
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE timestamp > ?', (time.time() - 86400,))
            recent_feedback = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_feedback': total_feedback,
                'feedback_by_type': feedback_by_type,
                'accuracy': accuracy,
                'recent_feedback_24h': recent_feedback,
                'feedback_queue_size': len(self.feedback_queue)
            }
            
        except Exception as e:
            logger.error(f"Error calculating feedback stats: {e}")
            return {}
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        # This would be populated by the continuous learning system
        return {
            'engagement_classifier': {
                'accuracy': 0.87,
                'precision': 0.85,
                'recall': 0.89,
                'f1_score': 0.87,
                'samples_trained': 1250
            },
            'behavioral_classifier': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84,
                'f1_score': 0.82,
                'samples_trained': 980
            },
            'emotion_classifier': {
                'accuracy': 0.79,
                'precision': 0.77,
                'recall': 0.81,
                'f1_score': 0.79,
                'samples_trained': 750
            }
        }
    
    def get_feedback_for_training(self, limit: int = 100) -> List[FeedbackEntry]:
        """Get feedback entries for training"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM feedback 
                WHERE actual_engagement IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            feedback_entries = []
            for row in rows:
                feedback_entry = FeedbackEntry(
                    timestamp=row[1],
                    session_id=row[2],
                    student_id=row[3],
                    predicted_engagement=row[4],
                    actual_engagement=row[5],
                    confidence=row[6],
                    feedback_type=row[7],
                    comments=row[8],
                    teacher_id=row[9],
                    context=json.loads(row[10]) if row[10] else {}
                )
                feedback_entries.append(feedback_entry)
            
            return feedback_entries
            
        except Exception as e:
            logger.error(f"Error getting feedback for training: {e}")
            return []
    
    def export_feedback_dataset(self, filepath: str):
        """Export feedback as training dataset"""
        try:
            feedback_entries = self.get_feedback_for_training(limit=10000)
            
            dataset = []
            for entry in feedback_entries:
                dataset.append({
                    'id': f"feedback_{int(entry.timestamp)}",
                    'features': entry.context.get('features', {}),
                    'predicted_label': entry.predicted_engagement,
                    'actual_label': entry.actual_engagement,
                    'confidence': entry.confidence,
                    'metadata': {
                        'source': 'teacher_feedback',
                        'feedback_type': entry.feedback_type,
                        'session_id': entry.session_id,
                        'teacher_id': entry.teacher_id,
                        'comments': entry.comments
                    }
                })
            
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Exported {len(dataset)} feedback entries to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting feedback dataset: {e}")

# HTML Template for feedback interface
FEEDBACK_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Engagement Feedback Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .prediction { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .high-engagement { border-left: 5px solid #4CAF50; }
        .medium-engagement { border-left: 5px solid #FF9800; }
        .low-engagement { border-left: 5px solid #F44336; }
        .feedback-form { background: #f9f9f9; padding: 15px; margin-top: 10px; border-radius: 5px; }
        button { background: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #45a049; }
        select, textarea { width: 100%; padding: 5px; margin: 5px 0; }
        .stats { background: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Real-time Engagement Feedback</h1>
    
    <div class="stats" id="stats">
        <h3>System Statistics</h3>
        <p>Loading...</p>
    </div>
    
    <div id="predictions">
        <h3>Current Predictions</h3>
        <p>Loading predictions...</p>
    </div>

    <script>
        function loadPredictions() {
            fetch('/api/current_predictions')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('predictions');
                    container.innerHTML = '<h3>Current Predictions</h3>';
                    
                    data.forEach(prediction => {
                        const div = document.createElement('div');
                        div.className = `prediction ${prediction.predicted_engagement.replace('_', '-')}`;
                        div.innerHTML = `
                            <h4>Student: ${prediction.student_id}</h4>
                            <p><strong>Predicted:</strong> ${prediction.predicted_engagement}</p>
                            <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%</p>
                            <div class="feedback-form">
                                <label>Actual Engagement:</label>
                                <select id="actual_${prediction.student_id}">
                                    <option value="">Select actual engagement...</option>
                                    <option value="high_engagement">High Engagement</option>
                                    <option value="medium_engagement">Medium Engagement</option>
                                    <option value="low_engagement">Low Engagement</option>
                                </select>
                                <label>Comments (optional):</label>
                                <textarea id="comments_${prediction.student_id}" rows="2" placeholder="Additional observations..."></textarea>
                                <button onclick="submitFeedback('${prediction.student_id}', '${prediction.predicted_engagement}', ${prediction.confidence})">
                                    Submit Feedback
                                </button>
                            </div>
                        `;
                        container.appendChild(div);
                    });
                })
                .catch(error => console.error('Error loading predictions:', error));
        }
        
        function submitFeedback(studentId, predicted, confidence) {
            const actual = document.getElementById(`actual_${studentId}`).value;
            const comments = document.getElementById(`comments_${studentId}`).value;
            
            if (!actual) {
                alert('Please select the actual engagement level');
                return;
            }
            
            const feedback = {
                student_id: studentId,
                predicted_engagement: predicted,
                actual_engagement: actual,
                confidence: confidence,
                comments: comments,
                feedback_type: 'teacher',
                teacher_id: 'teacher_1', // This would come from login
                session_id: 'current_session',
                context: {
                    timestamp: Date.now(),
                    interface: 'web'
                }
            };
            
            fetch('/api/submit_feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedback)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert('Feedback submitted successfully!');
                    loadPredictions(); // Reload predictions
                    loadStats(); // Reload stats
                } else {
                    alert('Error submitting feedback: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error submitting feedback:', error);
                alert('Error submitting feedback');
            });
        }
        
        function loadStats() {
            fetch('/api/feedback_stats')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('stats');
                    container.innerHTML = `
                        <h3>System Statistics</h3>
                        <p><strong>Total Feedback:</strong> ${data.total_feedback || 0}</p>
                        <p><strong>Model Accuracy:</strong> ${((data.accuracy || 0) * 100).toFixed(1)}%</p>
                        <p><strong>Recent Feedback (24h):</strong> ${data.recent_feedback_24h || 0}</p>
                        <p><strong>Pending Feedback:</strong> ${data.feedback_queue_size || 0}</p>
                    `;
                })
                .catch(error => console.error('Error loading stats:', error));
        }
        
        // Load data on page load
        loadPredictions();
        loadStats();
        
        // Refresh every 30 seconds
        setInterval(() => {
            loadPredictions();
            loadStats();
        }, 30000);
    </script>
</body>
</html>
'''

# Create templates directory and save template
def create_feedback_template():
    """Create feedback interface template"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    template_path = templates_dir / "feedback_interface.html"
    with open(template_path, 'w') as f:
        f.write(FEEDBACK_TEMPLATE)
    
    logger.info("Feedback interface template created")

if __name__ == "__main__":
    create_feedback_template()
    
    # Test feedback interface
    config = {'feedback_port': 5001}
    interface = FeedbackInterface(config)
    interface.start_server()
    
    print("Feedback interface running at http://127.0.0.1:5001")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        interface.stop_server()
