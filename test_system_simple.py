#!/usr/bin/env python3
"""
Simple System Test
Tests the attendance system without complex dependencies
"""

import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_files():
    """Test if dataset files exist and are properly formatted"""
    print("🧪 TESTING DATASET FILES")
    print("="*50)
    
    # Check CSV file
    csv_path = "data/student_dataset/student_metadata.csv"
    if os.path.exists(csv_path):
        print(f"✅ CSV file exists: {csv_path}")
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            print(f"   📊 Contains {len(lines)-1} student records")
            if len(lines) > 1:
                print(f"   📋 Header: {lines[0].strip()}")
                for i, line in enumerate(lines[1:], 1):
                    if line.strip():
                        print(f"   📝 Student {i}: {line.strip()}")
    else:
        print(f"❌ CSV file missing: {csv_path}")
        return False
    
    # Check JSON file
    json_path = "data/student_dataset/students.json"
    if os.path.exists(json_path):
        print(f"\n✅ JSON file exists: {json_path}")
        try:
            with open(json_path, 'r') as f:
                students = json.load(f)
                print(f"   📊 Contains {len(students)} students")
                for student in students:
                    name = student.get('name', 'Unknown')
                    roll = student.get('roll_number', 'Unknown')
                    photo = student.get('photo_path', 'Unknown')
                    print(f"   👤 {name} ({roll}) -> {photo}")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing error: {e}")
            return False
    else:
        print(f"❌ JSON file missing: {json_path}")
        return False
    
    # Check photo files
    print(f"\n📸 CHECKING PHOTO FILES:")
    photos_dir = "data/student_dataset/photos"
    if os.path.exists(photos_dir):
        photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   📁 Found {len(photo_files)} photo files")
        for photo_file in photo_files:
            photo_path = os.path.join(photos_dir, photo_file)
            file_size = os.path.getsize(photo_path)
            if file_size > 0:
                print(f"   ✅ {photo_file} ({file_size} bytes)")
            else:
                print(f"   ⚠️  {photo_file} (empty placeholder)")
    else:
        print(f"   ❌ Photos directory missing: {photos_dir}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print(f"\n🧪 TESTING CONFIGURATION")
    print("="*50)
    
    try:
        from config.config import config
        print("✅ Configuration loaded successfully")
        
        # Test attendance config
        if hasattr(config, 'attendance'):
            print("✅ Attendance configuration available")
            print(f"   📁 Dataset path: {config.attendance.student_dataset_path}")
            print(f"   🗄️  Database path: {config.attendance.attendance_db_path}")
            print(f"   🎯 Recognition threshold: {config.attendance.face_recognition_threshold}")
            print(f"   ⏰ Alert duration: {config.attendance.disappearance_alert_duration}s")
        else:
            print("❌ Attendance configuration missing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def test_database_creation():
    """Test database creation"""
    print(f"\n🧪 TESTING DATABASE CREATION")
    print("="*50)
    
    try:
        import sqlite3
        
        # Create test database
        db_path = "data/test_attendance.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_attendance (
                id INTEGER PRIMARY KEY,
                roll_number TEXT,
                name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert test record
        cursor.execute('''
            INSERT INTO test_attendance (roll_number, name) VALUES (?, ?)
        ''', ("TEST001", "Test Student"))
        
        # Query test record
        cursor.execute("SELECT * FROM test_attendance WHERE roll_number = ?", ("TEST001",))
        result = cursor.fetchone()
        
        if result:
            print("✅ Database operations working")
            print(f"   📝 Test record: {result}")
        else:
            print("❌ Database query failed")
            return False
        
        # Cleanup
        cursor.execute("DROP TABLE test_attendance")
        conn.commit()
        conn.close()
        os.remove(db_path)
        
        print("✅ Database test completed and cleaned up")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print(f"\n🧪 TESTING OPENCV")
    print("="*50)
    
    try:
        import cv2
        import numpy as np
        
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test basic operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        if gray.shape == (100, 100):
            print("✅ Basic image operations working")
        else:
            print("❌ Image operations failed")
            return False
        
        # Test camera availability (without opening)
        print("📹 Camera availability check...")
        # Note: We don't actually open the camera to avoid blocking
        print("   ℹ️  Camera test skipped (would require user interaction)")
        
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def print_face_recognition_status():
    """Check face recognition availability"""
    print(f"\n🧪 FACE RECOGNITION STATUS")
    print("="*50)
    
    try:
        import face_recognition
        print("✅ face_recognition library available")
        print("✅ Full facial recognition functionality enabled")
        return True
    except ImportError:
        print("⚠️  face_recognition library not available")
        print("📋 INSTALLATION INSTRUCTIONS:")
        print("   1. Install CMake: pip install cmake")
        print("   2. Install dlib: pip install dlib")
        print("   3. Install face_recognition: pip install face-recognition")
        print("   ")
        print("🔄 ALTERNATIVE (if above fails):")
        print("   1. Download Visual Studio Build Tools")
        print("   2. Or use conda: conda install -c conda-forge dlib")
        print("   3. Then: pip install face-recognition")
        print("   ")
        print("⚡ CURRENT STATUS:")
        print("   - System will work in detection-only mode")
        print("   - Face detection and tracking: ✅ Working")
        print("   - Face recognition: ❌ Disabled")
        print("   - Attendance logging: ⚠️  Limited (unknown persons only)")
        return False

def main():
    """Main test function"""
    print("🎓 AUTOMATED ATTENDANCE SYSTEM - SIMPLE TEST")
    print("="*60)
    print("Testing core system components")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Dataset files
    if not test_dataset_files():
        all_tests_passed = False
    
    # Test 2: Configuration
    if not test_config_loading():
        all_tests_passed = False
    
    # Test 3: Database
    if not test_database_creation():
        all_tests_passed = False
    
    # Test 4: OpenCV
    if not test_opencv():
        all_tests_passed = False
    
    # Test 5: Face recognition status
    face_recognition_available = print_face_recognition_status()
    
    # Final summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if all_tests_passed:
        print("✅ CORE SYSTEM TESTS PASSED!")
        print("\n📋 SYSTEM STATUS:")
        print("✅ Dataset structure: Working")
        print("✅ Configuration: Working") 
        print("✅ Database operations: Working")
        print("✅ OpenCV: Working")
        
        if face_recognition_available:
            print("✅ Face recognition: Available")
            print("\n🚀 READY FOR FULL DEPLOYMENT!")
            print("   - Run: python src/main.py")
            print("   - Or test: python test_attendance_system.py")
        else:
            print("⚠️  Face recognition: Not available")
            print("\n🔧 NEXT STEPS:")
            print("   1. Install face_recognition library (see instructions above)")
            print("   2. Replace placeholder photos with real student photos")
            print("   3. Run: python src/main.py")
        
        print("\n📸 PHOTO SETUP:")
        print("   - Replace placeholder photos in data/student_dataset/photos/")
        print("   - Use clear, well-lit photos of students")
        print("   - Ensure filenames match roll numbers")
        
    else:
        print("❌ SOME CORE TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
    
    print("="*60)

if __name__ == "__main__":
    main()
