#!/usr/bin/env python3
"""
Demo Script for Real-time Classroom Engagement Analyzer
Quick start script for hackathon demonstration
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

def print_banner():
    """Print demo banner"""
    print("🎓" + "=" * 58 + "🎓")
    print("   REAL-TIME CLASSROOM ENGAGEMENT ANALYZER DEMO")
    print("        Hackathon Project by Subhasis & Sachin")
    print("🎓" + "=" * 58 + "🎓")
    print()

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'ultralytics', 'mediapipe', 
        'librosa', 'pyaudio', 'textblob', 'websocket'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def check_camera():
    """Check if camera is available"""
    print("\n📹 Checking camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not available")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            cap.release()
            return False
        
        print(f"✅ Camera available: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def check_audio():
    """Check if audio is available"""
    print("\n🎤 Checking audio...")
    
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        # Check for input devices
        input_devices = []
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
        
        audio.terminate()
        
        if input_devices:
            print(f"✅ Audio input available: {len(input_devices)} devices")
            return True
        else:
            print("❌ No audio input devices found")
            return False
            
    except Exception as e:
        print(f"❌ Audio check failed: {e}")
        return False

def run_tests():
    """Run quick system tests"""
    print("\n🧪 Running system tests...")
    
    try:
        # Run basic functionality tests
        result = subprocess.run([
            sys.executable, 'tests/test_pipeline.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ System tests passed!")
            return True
        else:
            print("❌ System tests failed!")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def start_mock_backend():
    """Start a mock backend server for demo"""
    print("\n🖥️  Starting mock backend server...")
    
    try:
        from flask import Flask, request, jsonify
        from flask_socketio import SocketIO, emit
        import threading
        
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'demo_secret'
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        # Store engagement data
        engagement_data = []
        
        @app.route('/api/health')
        def health():
            return jsonify({'status': 'ok', 'service': 'engagement_analyzer_demo'})
        
        @app.route('/api/engagement', methods=['POST'])
        def receive_engagement():
            data = request.json
            engagement_data.append(data)
            
            # Keep only last 100 entries
            if len(engagement_data) > 100:
                engagement_data.pop(0)
            
            return jsonify({'status': 'received'})
        
        @app.route('/api/engagement', methods=['GET'])
        def get_engagement():
            return jsonify(engagement_data[-10:])  # Last 10 entries
        
        @socketio.on('connect')
        def handle_connect():
            print(f"   📡 Client connected: {request.sid}")
            emit('status', {'message': 'Connected to demo backend'})
        
        @socketio.on('disconnect')
        def handle_disconnect():
            print(f"   📡 Client disconnected: {request.sid}")
        
        def run_server():
            socketio.run(app, host='localhost', port=3000, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        print("✅ Mock backend started on http://localhost:3000")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start mock backend: {e}")
        return False

def run_engagement_analyzer():
    """Run the main engagement analyzer"""
    print("\n🚀 Starting Engagement Analyzer...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Import and run the main application
        from src.main import main
        main()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

def main():
    """Main demo function"""
    print_banner()
    
    # System checks
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install requirements.")
        sys.exit(1)
    
    if not check_camera():
        print("\n⚠️  Camera not available. Demo will run with limited functionality.")
    
    if not check_audio():
        print("\n⚠️  Audio not available. Demo will run without audio analysis.")
    
    # Optional: Run tests
    print("\n" + "=" * 60)
    run_tests_choice = input("🧪 Run system tests? (y/N): ").lower().strip()
    
    if run_tests_choice == 'y':
        if not run_tests():
            print("\n⚠️  Tests failed, but continuing with demo...")
    
    # Start backend
    print("\n" + "=" * 60)
    start_backend_choice = input("🖥️  Start mock backend server? (Y/n): ").lower().strip()
    
    if start_backend_choice != 'n':
        if not start_mock_backend():
            print("\n⚠️  Backend failed to start, continuing without backend...")
    
    # Start main application
    print("\n" + "=" * 60)
    print("🎯 DEMO READY!")
    print("\nThe engagement analyzer will:")
    print("  • Detect faces and count attendance")
    print("  • Analyze head poses for attention")
    print("  • Recognize participation gestures")
    print("  • Process audio for engagement")
    print("  • Calculate real-time engagement scores")
    print("  • Send data to backend (if available)")
    
    input("\nPress Enter to start the demo...")
    
    run_engagement_analyzer()
    
    print("\n🎉 Demo completed!")
    print("Thank you for trying the Real-time Classroom Engagement Analyzer!")

if __name__ == "__main__":
    main()
