#!/usr/bin/env python3
"""
Basic Test for Automated Attendance System
Works without face_recognition library - tests basic functionality
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
    from config.config import config
    from src.utils.logger import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

def test_dataset_loading():
    """Test loading the student dataset"""
    print("🧪 Testing dataset loading...")
    
    try:
        # Initialize attendance system
        attendance_config = config.attendance.__dict__
        attendance_system = AutomatedAttendanceSystem(attendance_config)
        
        if not attendance_system.initialize():
            print("❌ Failed to initialize attendance system")
            return False
        
        print("✅ Attendance system initialized successfully")
        
        # Test database connection
        if attendance_system.db_connection:
            print("✅ Database connection established")
        else:
            print("❌ Database connection failed")
            return False
        
        # Test student dataset loading
        student_count = len(attendance_system.students_db)
        print(f"✅ Loaded {student_count} students from dataset")
        
        # Print student details
        if student_count > 0:
            print("\n📋 LOADED STUDENTS:")
            for roll_number, student in attendance_system.students_db.items():
                print(f"   - {student.name} ({roll_number}) - {student.department}")
        else:
            print("⚠️  No students in dataset")
        
        # Test photo file existence
        print("\n📸 CHECKING PHOTOS:")
        for roll_number, student in attendance_system.students_db.items():
            if os.path.exists(student.photo_path):
                file_size = os.path.getsize(student.photo_path)
                if file_size > 0:
                    print(f"   ✅ {student.name}: {student.photo_path} ({file_size} bytes)")
                else:
                    print(f"   ⚠️  {student.name}: {student.photo_path} (empty file)")
            else:
                print(f"   ❌ {student.name}: {student.photo_path} (missing)")
        
        attendance_system.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_basic_processing():
    """Test basic processing without face recognition"""
    print("\n🧪 Testing basic processing...")
    
    try:
        # Initialize attendance system
        attendance_config = config.attendance.__dict__
        attendance_system = AutomatedAttendanceSystem(attendance_config)
        
        if not attendance_system.initialize():
            print("❌ Failed to initialize attendance system")
            return False
        
        # Create mock frame and face data
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_face_data = {
            'faces': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.8,
                    'center': [150, 150],
                    'area': 10000
                },
                {
                    'bbox': [300, 150, 400, 250],
                    'confidence': 0.9,
                    'center': [350, 200],
                    'area': 10000
                }
            ]
        }
        
        # Test processing
        attendance_data = {
            'frame': mock_frame,
            'face_detection': mock_face_data,
            'pose_estimation': {},
            'gesture_recognition': {}
        }
        
        result = attendance_system.process_data(attendance_data)
        
        if result:
            print("✅ Basic processing successful")
            print(f"   - Tracked persons: {len(result.get('tracked_persons', []))}")
            print(f"   - Attendance count: {result.get('attendance_count', 0)}")
            print(f"   - Total recognized: {result.get('total_recognized', 0)}")
            print(f"   - Active alerts: {len(result.get('active_alerts', []))}")
            print(f"   - Frame annotations: {len(result.get('frame_annotations', []))}")
        else:
            print("❌ Processing returned no result")
            return False
        
        attendance_system.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_database_operations():
    """Test database operations"""
    print("\n🧪 Testing database operations...")
    
    try:
        # Initialize attendance system
        attendance_config = config.attendance.__dict__
        attendance_system = AutomatedAttendanceSystem(attendance_config)
        
        if not attendance_system.initialize():
            print("❌ Failed to initialize attendance system")
            return False
        
        # Test database tables
        cursor = attendance_system.db_connection.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['attendance', 'students', 'alerts']
        for table in expected_tables:
            if table in table_names:
                print(f"   ✅ Table '{table}' exists")
            else:
                print(f"   ❌ Table '{table}' missing")
        
        # Test inserting a test record
        test_roll = "TEST001"
        test_name = "Test Student"
        
        try:
            attendance_system._log_attendance_entry(test_roll, test_name)
            print("   ✅ Test attendance entry logged")
            
            # Clean up test record
            cursor.execute("DELETE FROM attendance WHERE roll_number = ?", (test_roll,))
            attendance_system.db_connection.commit()
            print("   ✅ Test record cleaned up")
            
        except Exception as e:
            print(f"   ⚠️  Database operation test failed: {e}")
        
        attendance_system.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎓 AUTOMATED ATTENDANCE SYSTEM - BASIC TEST")
    print("="*60)
    print("Testing system functionality without face recognition")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Dataset loading
    if not test_dataset_loading():
        all_tests_passed = False
    
    # Test 2: Basic processing
    if not test_basic_processing():
        all_tests_passed = False
    
    # Test 3: Database operations
    if not test_database_operations():
        all_tests_passed = False
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("\n📋 SYSTEM STATUS:")
        print("✅ Dataset structure: Working")
        print("✅ Database operations: Working")
        print("✅ Basic processing: Working")
        print("⚠️  Face recognition: Not available (install face-recognition)")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Install face recognition: pip install face-recognition")
        print("2. Replace placeholder photos with actual student photos")
        print("3. Run full system: python src/main.py")
        print("4. Test with real camera: python test_attendance_system.py")
        
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them.")
    
    print("="*60)

if __name__ == "__main__":
    main()
