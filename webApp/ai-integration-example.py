#!/usr/bin/env python3
"""
AI Pipeline Integration Example for Classroom Engagement Analyzer
This script shows how to send AI-processed data to the backend system.

For Subhasis: Use this as a template for your AI pipeline integration.
"""

import requests
import json
import time
import random
from datetime import datetime

# Backend API configuration
API_BASE_URL = "http://localhost:3001/api"
SESSION_ID = "cmda2lp3s0000oitg94iw420n"  # Replace with actual session ID

def send_metrics_to_backend(session_id, metrics_data):
    """
    Send AI-processed metrics to the backend system
    
    Args:
        session_id (str): The session ID to send metrics for
        metrics_data (dict): The processed metrics from AI pipeline
    
    Returns:
        dict: Response from the backend
    """
    url = f"{API_BASE_URL}/ai/metrics"
    
    payload = {
        "sessionId": session_id,
        "metrics": metrics_data,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending metrics: {e}")
        return None

def simulate_ai_processing():
    """
    Simulate AI processing results - replace this with your actual AI pipeline
    
    Returns:
        dict: Simulated metrics data in the expected format
    """
    # Simulate some variation in metrics
    base_engagement = random.uniform(0.6, 0.9)
    noise_variation = random.uniform(-0.1, 0.1)
    
    return {
        "attendance": {
            "total_detected": random.randint(20, 28),
            "confidence_avg": random.uniform(0.85, 0.95)
        },
        "engagement": {
            "overall_score": max(0.0, min(1.0, base_engagement + noise_variation)),
            "attention_score": max(0.0, min(1.0, base_engagement + random.uniform(-0.1, 0.1))),
            "participation_score": max(0.0, min(1.0, base_engagement + random.uniform(-0.15, 0.1))),
            "zones": {
                "front": max(0.0, min(1.0, base_engagement + random.uniform(0.0, 0.2))),
                "middle": max(0.0, min(1.0, base_engagement + random.uniform(-0.1, 0.1))),
                "back": max(0.0, min(1.0, base_engagement + random.uniform(-0.2, 0.0)))
            }
        },
        "audio": {
            "noise_level": random.uniform(0.2, 0.8),
            "speaker_activity": random.uniform(0.5, 0.9)
        }
    }

def create_test_session():
    """Create a test session for demo purposes"""
    url = f"{API_BASE_URL}/sessions"
    
    payload = {
        "name": "AI Demo Session",
        "description": "Demo session for AI pipeline integration",
        "totalStudents": 25
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        session_data = response.json()
        return session_data["data"]["session"]["id"]
    except requests.exceptions.RequestException as e:
        print(f"Error creating session: {e}")
        return None

def main():
    """
    Main function - demonstrates how to integrate AI pipeline with backend
    """
    print("🤖 AI Pipeline Integration Demo")
    print("=" * 50)
    
    # Option 1: Create a new session
    print("Creating new demo session...")
    session_id = create_test_session()
    
    if not session_id:
        print("❌ Failed to create session, using default session ID")
        session_id = SESSION_ID
    else:
        print(f"✅ Created session: {session_id}")
    
    print(f"\n📊 Dashboard URL: http://localhost:5173")
    print(f"🔗 Join the session in the dashboard to see real-time updates!")
    print("\n🚀 Starting AI pipeline simulation...")
    print("Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Simulate AI processing (replace with your actual AI pipeline)
            print(f"[{iteration:03d}] Processing frame...")
            
            # Your AI pipeline would process camera/audio input here
            # For demo, we simulate the results
            metrics = simulate_ai_processing()
            
            # Send metrics to backend
            result = send_metrics_to_backend(session_id, metrics)
            
            if result and result.get("success"):
                print(f"✅ Metrics sent successfully")
                print(f"   Attendance: {metrics['attendance']['total_detected']} students")
                print(f"   Engagement: {metrics['engagement']['overall_score']:.1%}")
                print(f"   Zones: F:{metrics['engagement']['zones']['front']:.1%} "
                      f"M:{metrics['engagement']['zones']['middle']:.1%} "
                      f"B:{metrics['engagement']['zones']['back']:.1%}")
                
                # Check if engagement is low (will trigger alert)
                if metrics['engagement']['overall_score'] < 0.6:
                    print("⚠️  Low engagement detected - alert should be triggered!")
                
                if metrics['audio']['noise_level'] > 0.7:
                    print("🔊 High noise level detected - alert should be triggered!")
                    
            else:
                print("❌ Failed to send metrics")
            
            print("-" * 50)
            
            # Wait before next iteration (adjust based on your AI processing speed)
            time.sleep(5)  # 5 seconds between updates
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping AI pipeline simulation...")
        print("Demo completed!")

if __name__ == "__main__":
    # Check if backend is running
    try:
        response = requests.get(f"{API_BASE_URL}/../health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running")
            main()
        else:
            print("❌ Backend server is not responding correctly")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to backend server")
        print("Make sure the backend is running on http://localhost:3001")
        print("Run: node src/server/simple-server.js")
