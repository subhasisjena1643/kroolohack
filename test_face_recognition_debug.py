#!/usr/bin/env python3
"""
Debug Face Recognition Test
Test facial recognition with your actual photos and live camera
"""

import os
import sys
import cv2
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_face_recognition_with_camera():
    """Test face recognition with live camera"""
    print("🎓 TESTING FACE RECOGNITION WITH LIVE CAMERA")
    print("="*60)
    
    try:
        from src.modules.automated_attendance_system import AutomatedAttendanceSystem
        from config.config import config
        from src.utils.logger import logger
        
        # Initialize attendance system
        attendance_config = config.attendance.__dict__
        attendance_system = AutomatedAttendanceSystem(attendance_config)
        
        if not attendance_system.initialize():
            print("❌ Failed to initialize attendance system")
            return False
        
        print(f"✅ Attendance system initialized")
        print(f"   📊 Students loaded: {len(attendance_system.students_db)}")
        print(f"   🧠 Face encodings: {len(attendance_system.known_face_encodings)}")
        print(f"   🎯 Recognition threshold: {attendance_system.face_recognition_threshold}")
        
        # List loaded students
        print("\n👥 LOADED STUDENTS:")
        for roll_number, student in attendance_system.students_db.items():
            print(f"   - {student.name} ({roll_number})")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        print("\n📹 Camera opened successfully")
        print("🔍 Starting face recognition test...")
        print("Press 'q' to quit, 's' to show stats")
        
        frame_count = 0
        recognition_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Simple face detection using OpenCV (for testing)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Convert to our format
            face_data = []
            for (x, y, w, h) in faces:
                face_data.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.8,
                    'center': [x+w//2, y+h//2],
                    'area': w*h
                })
            
            # Test attendance system processing
            if len(face_data) > 0 and frame_count % 30 == 0:  # Test every 30 frames
                print(f"\n🔍 Frame {frame_count}: Testing {len(face_data)} detected faces...")
                
                attendance_data = {
                    'frame': frame,
                    'face_detection': {'faces': face_data},
                    'pose_estimation': {},
                    'gesture_recognition': {}
                }
                
                result = attendance_system.process_data(attendance_data)
                
                if result:
                    recognized = result.get('total_recognized', 0)
                    tracked = len(result.get('tracked_persons', []))
                    alerts = len(result.get('active_alerts', []))
                    
                    print(f"   📊 Results: {recognized} recognized, {tracked} tracked, {alerts} alerts")
                    
                    if recognized > 0:
                        recognition_count += 1
                        print("   🎉 RECOGNITION SUCCESS!")
                    
                    # Show tracked persons
                    for person in result.get('tracked_persons', []):
                        name = person.get('name', 'Unknown')
                        roll = person.get('roll_number', 'N/A')
                        confidence = person.get('confidence', 0)
                        print(f"   👤 {name} ({roll}) - Confidence: {confidence:.3f}")
            
            # Draw face rectangles
            for face in face_data:
                bbox = face['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {face['confidence']:.2f}", 
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show stats on frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(face_data)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Recognitions: {recognition_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition Debug', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n📊 STATS:")
                print(f"   Frames processed: {frame_count}")
                print(f"   Recognition attempts: {recognition_count}")
                print(f"   Current faces detected: {len(face_data)}")
        
        cap.release()
        cv2.destroyAllWindows()
        attendance_system.cleanup()
        
        print(f"\n✅ Test completed!")
        print(f"   Total frames: {frame_count}")
        print(f"   Recognition successes: {recognition_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_photo_recognition():
    """Test recognition directly with stored photos"""
    print("\n🧪 TESTING DIRECT PHOTO RECOGNITION")
    print("="*60)
    
    try:
        from deepface import DeepFace
        import json
        
        # Load student data
        with open("data/student_dataset/students.json", 'r') as f:
            students = json.load(f)
        
        print(f"Testing recognition between {len(students)} student photos...")
        
        # Test recognition between different students
        for i, student1 in enumerate(students):
            for j, student2 in enumerate(students):
                if i >= j:  # Skip same and already tested pairs
                    continue
                
                photo1 = student1['photo_path']
                photo2 = student2['photo_path']
                name1 = student1['name']
                name2 = student2['name']
                
                if os.path.exists(photo1) and os.path.exists(photo2):
                    try:
                        result = DeepFace.verify(
                            img1_path=photo1,
                            img2_path=photo2,
                            model_name='Facenet',
                            detector_backend='opencv'
                        )
                        
                        distance = result['distance']
                        threshold = result['threshold']
                        verified = result['verified']
                        
                        print(f"   {name1} vs {name2}:")
                        print(f"     Distance: {distance:.4f} | Threshold: {threshold:.4f} | Same: {verified}")
                        
                    except Exception as e:
                        print(f"   {name1} vs {name2}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct photo test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎓 FACE RECOGNITION DEBUG TEST")
    print("="*60)
    print("Testing facial recognition with your student photos")
    print("="*60)
    
    # Test 1: Direct photo recognition
    if not test_direct_photo_recognition():
        print("❌ Direct photo recognition failed")
        return
    
    # Test 2: Live camera recognition
    print("\n" + "="*60)
    response = input("Test with live camera? (y/n): ").lower().strip()
    if response == 'y':
        if not test_face_recognition_with_camera():
            print("❌ Live camera recognition failed")
            return
    
    print("\n" + "="*60)
    print("🎉 FACE RECOGNITION DEBUG COMPLETE!")
    print("="*60)
    
    print("\n📋 TROUBLESHOOTING TIPS:")
    print("1. If no faces are recognized:")
    print("   - Check if photos are clear and well-lit")
    print("   - Lower the recognition threshold in config")
    print("   - Ensure good lighting during camera test")
    
    print("\n2. If alerts aren't working:")
    print("   - Make sure persons are being tracked first")
    print("   - Check the 5-second grace period")
    print("   - Look for alert messages in the logs")
    
    print("\n3. If performance is slow:")
    print("   - Recognition happens every frame now (for testing)")
    print("   - DeepFace processing takes time initially")
    print("   - Subsequent recognitions should be faster")

if __name__ == "__main__":
    main()
