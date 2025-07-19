#!/usr/bin/env python3
"""
Test the web interface separately
"""

import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.modules.feedback_interface import FeedbackInterface
    
    print("🌐 Testing Feedback Interface...")
    
    # Create config
    config = {
        'feedback_port': 5001,
        'feedback_host': '127.0.0.1'
    }
    
    # Create and start feedback interface
    feedback = FeedbackInterface(config)
    feedback.start_server()
    
    print("✅ Feedback interface started!")
    print("🌐 Open: http://127.0.0.1:5001")
    print("⏹️  Press Ctrl+C to stop")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        feedback.stop_server()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
