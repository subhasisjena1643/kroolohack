#!/usr/bin/env python3
"""
Test script for the Automated Attendance System
Demonstrates facial recognition, tracking, and attendance logging
"""

import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.modules.automated_attendance_system import AutomatedAttendanceSystem
    from utils.dataset_manager import StudentDatasetManager
    from config.config import config
    from utils.logger import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install face-recognition dlib")
    sys.exit(1)

def create_sample_dataset():
    """Create a sample student dataset for testing"""
    print("Creating sample student dataset...")
    
    # Create dataset manager
    manager = StudentDatasetManager()
    
    # Sample students (you would replace these with real photos)
    sample_students = [
        {
            "roll_number": "CS2021001",
            "name": "Test Student 1",
            "application_number": "APP2021001",
            "department": "Computer Science",
            "year": "2021",
            "section": "A"
        },
        {
            "roll_number": "CS2021002", 
            "name": "Test Student 2",
            "application_number": "APP2021002",
            "department": "Computer Science",
            "year": "2021",
            "section": "A"
        }
    ]
    
    # Note: In real usage, you would have actual student photos
    print("Note: This is a demo. In real usage, add student photos using:")
    print("python utils/dataset_manager.py --add --roll CS2021001 --name 'Student Name' --app-num APP2021001 --photo path/to/photo.jpg")
    
    return True

def test_attendance_system():
    """Test the automated attendance system"""
    print("Testing Automated Attendance System...")
    
    try:
        # Initialize attendance system
        attendance_config = config.attendance.__dict__
        attendance_system = AutomatedAttendanceSystem(attendance_config)
        
        if not attendance_system.initialize():
            print("Failed to initialize attendance system")
            return False
        
        print("✓ Attendance system initialized successfully")
        
        # Test database connection
        if attendance_system.db_connection:
            print("✓ Database connection established")
        else:
            print("✗ Database connection failed")
            return False
        
        # Test student dataset loading
        student_count = len(attendance_system.students_db)
        print(f"✓ Loaded {student_count} students from dataset")
        
        if student_count == 0:
            print("Note: No students in dataset. Add students using dataset_manager.py")
        
        # Test camera and processing
        print("Testing live camera processing...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Cannot open camera")
            return False
        
        print("✓ Camera opened successfully")
        print("Press 'q' to quit, 's' to show attendance summary")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                # Simulate face detection data (in real system, this comes from face detector)
                mock_face_data = {
                    'faces': [
                        {
                            'bbox': [100, 100, 200, 200],
                            'confidence': 0.8,
                            'center': [150, 150],
                            'area': 10000
                        }
                    ]
                }
                
                # Process attendance data
                attendance_data = {
                    'frame': frame,
                    'face_detection': mock_face_data,
                    'pose_estimation': {},
                    'gesture_recognition': {}
                }
                
                result = attendance_system.process_data(attendance_data)
                
                # Display results on frame
                if result:
                    # Draw attendance info
                    y_offset = 30
                    cv2.putText(frame, f"Attendance System Active", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                    
                    cv2.putText(frame, f"Present: {result.get('attendance_count', 0)}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                    
                    cv2.putText(frame, f"Recognized: {result.get('total_recognized', 0)}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                    
                    cv2.putText(frame, f"Alerts: {len(result.get('active_alerts', []))}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw face annotations
                    for annotation in result.get('frame_annotations', []):
                        if annotation['type'] == 'rectangle':
                            bbox = annotation['bbox']
                            color = annotation['color']
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                            cv2.putText(frame, annotation['label'], (bbox[0], bbox[1]-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Automated Attendance System Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show attendance summary
                summary = attendance_system.get_attendance_summary()
                print("\n" + "="*50)
                print("ATTENDANCE SUMMARY")
                print("="*50)
                print(f"Present Students: {summary['present_students']}")
                print(f"Unknown Persons: {summary['unknown_persons']}")
                print(f"Total Recognized: {summary['total_recognized']}")
                print(f"Active Alerts: {summary['active_alerts']}")
                print("="*50 + "\n")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        attendance_system.cleanup()
        
        print("✓ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎓 AUTOMATED ATTENDANCE SYSTEM TEST")
    print("="*50)
    
    # Create sample dataset
    if not create_sample_dataset():
        print("Failed to create sample dataset")
        return
    
    # Test attendance system
    if not test_attendance_system():
        print("Attendance system test failed")
        return
    
    print("\n✅ All tests completed!")
    print("\nTo use the attendance system in production:")
    print("1. Add student photos using: python utils/dataset_manager.py")
    print("2. Run the main application: python src/main.py")
    print("3. Students will be automatically recognized and tracked")
    print("4. Attendance records are saved to the database")
    print("5. Alerts are generated for disappearances")

if __name__ == "__main__":
    main()
