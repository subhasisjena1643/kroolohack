#!/usr/bin/env python3
"""
College Dataset Setup Script
Bulk import student data from photos and CSV metadata
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.dataset_manager import StudentDatasetManager
    from config.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

def setup_sample_photos():
    """Create sample photo placeholders for testing"""
    photos_dir = Path("data/student_dataset/photos")
    photos_dir.mkdir(parents=True, exist_ok=True)
    
    sample_files = [
        "CS2021001.jpg",
        "CS2021002.jpg", 
        "CS2021003.jpg"
    ]
    
    print("📸 Setting up sample photo placeholders...")
    for filename in sample_files:
        photo_path = photos_dir / filename
        if not photo_path.exists():
            # Create a placeholder file
            photo_path.touch()
            print(f"📁 Created placeholder: {filename}")
            print(f"   ⚠️  Replace with actual student photo!")
    
    return str(photos_dir)

def validate_setup():
    """Validate the dataset setup"""
    print("\n🔍 VALIDATING DATASET SETUP...")
    
    # Check required files
    required_files = [
        "data/student_dataset/student_metadata.csv",
        "data/student_dataset/photos/CS2021001.jpg",
        "data/student_dataset/photos/CS2021002.jpg",
        "data/student_dataset/photos/CS2021003.jpg"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 0:
                print(f"✅ {file_path} (size: {size} bytes)")
            else:
                print(f"⚠️  {file_path} (empty file - needs actual photo)")
                all_good = False
        else:
            print(f"❌ {file_path} (missing)")
            all_good = False
    
    return all_good

def run_bulk_import():
    """Run the bulk import process"""
    print("\n🚀 STARTING BULK IMPORT...")
    
    # Initialize dataset manager
    manager = StudentDatasetManager()
    
    # Paths
    photos_folder = "data/student_dataset/photos"
    csv_metadata = "data/student_dataset/student_metadata.csv"
    
    # Validate paths exist
    if not os.path.exists(photos_folder):
        print(f"❌ Photos folder not found: {photos_folder}")
        return False
    
    if not os.path.exists(csv_metadata):
        print(f"❌ CSV metadata file not found: {csv_metadata}")
        return False
    
    # Run bulk import
    imported_count = manager.bulk_import_from_folder(photos_folder, csv_metadata)
    
    if imported_count > 0:
        print(f"\n✅ Successfully imported {imported_count} students!")
        
        # Generate face encodings
        print("\n🧠 GENERATING FACE ENCODINGS...")
        if manager.generate_face_encodings():
            print("✅ Face encodings generated successfully!")
        else:
            print("⚠️  Face encoding generation had issues (check photo quality)")
        
        # Validate final dataset
        print("\n📊 FINAL DATASET VALIDATION...")
        stats = manager.validate_dataset()
        print(f"Total Students: {stats['total_students']}")
        print(f"Valid Photos: {stats['valid_photos']}")
        print(f"Valid Encodings: {stats['valid_encodings']}")
        
        if stats['issues']:
            print("\n⚠️  Issues found:")
            for issue in stats['issues']:
                print(f"   - {issue}")
        
        return True
    else:
        print("❌ No students were imported. Check your photos and CSV file.")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎓 COLLEGE DATASET SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Replace placeholder photos with actual student photos:")
    print("   - Copy real student photos to data/student_dataset/photos/")
    print("   - Ensure filenames match roll numbers (e.g., CS2021001.jpg)")
    print("   - Photos should be clear, well-lit, front-facing")
    
    print("\n2. Add more students to your dataset:")
    print("   - Add rows to data/student_dataset/student_metadata.csv")
    print("   - Add corresponding photos to photos/ folder")
    print("   - Run this script again to import new students")
    
    print("\n3. Test the attendance system:")
    print("   - Run: python test_attendance_system.py")
    print("   - Or run full system: python src/main.py")
    
    print("\n4. For large datasets (100+ students):")
    print("   - Prepare all photos in a folder")
    print("   - Create comprehensive CSV with all student data")
    print("   - Run bulk import for entire college database")
    
    print("\n📊 DATASET EXPANSION:")
    print("   - Current: 3 students (sample)")
    print("   - Scalable to: 1000+ students")
    print("   - Just add more rows to CSV and photos to folder!")

def main():
    """Main setup function"""
    print("🎓 COLLEGE DATASET SETUP - METHOD 3 (BULK IMPORT)")
    print("="*60)
    
    # Step 1: Setup directory structure and sample photos
    photos_dir = setup_sample_photos()
    
    # Step 2: Validate setup
    if not validate_setup():
        print("\n❌ Setup validation failed. Please check the issues above.")
        return
    
    # Step 3: Ask user about photos
    print("\n📸 PHOTO SETUP:")
    print("Sample placeholder photos have been created.")
    print("You need to replace them with actual student photos.")
    
    response = input("\nDo you have actual student photos ready? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n✅ Great! Make sure your photos are:")
        print("   - Named with roll numbers (CS2021001.jpg, etc.)")
        print("   - Clear and well-lit")
        print("   - Front-facing or slight angle")
        print("   - Minimum 300x300 pixels")
        
        proceed = input("\nProceed with bulk import? (y/n): ").lower().strip()
        if proceed == 'y':
            # Step 4: Run bulk import
            if run_bulk_import():
                print_next_steps()
            else:
                print("\n❌ Bulk import failed. Check your photos and try again.")
        else:
            print("\n⏸️  Setup paused. Run this script again when ready.")
    else:
        print("\n📋 TO COMPLETE SETUP:")
        print("1. Replace placeholder photos in data/student_dataset/photos/")
        print("2. Ensure photos are named with roll numbers")
        print("3. Run this script again")
        print("\n💡 TIP: You can test with any 3 photos for now!")

if __name__ == "__main__":
    main()
