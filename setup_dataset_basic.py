#!/usr/bin/env python3
"""
Basic College Dataset Setup Script
Sets up the dataset structure without face recognition dependencies
"""

import os
import csv
import json
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data/student_dataset",
        "data/student_dataset/photos",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_csv_metadata():
    """Create the CSV metadata file"""
    print("\n📊 Creating CSV metadata file...")
    
    csv_path = "data/student_dataset/student_metadata.csv"
    
    # Sample data for 3 students
    students_data = [
        {
            "filename": "CS2021001.jpg",
            "roll_number": "CS2021001", 
            "name": "Aarav Sharma",
            "application_number": "APP2021001",
            "department": "Computer Science Engineering",
            "year": "2021",
            "section": "A"
        },
        {
            "filename": "CS2021002.jpg",
            "roll_number": "CS2021002",
            "name": "Priya Patel", 
            "application_number": "APP2021002",
            "department": "Computer Science Engineering",
            "year": "2021",
            "section": "A"
        },
        {
            "filename": "CS2021003.jpg",
            "roll_number": "CS2021003",
            "name": "Rahul Kumar",
            "application_number": "APP2021003", 
            "department": "Computer Science Engineering",
            "year": "2021",
            "section": "A"
        }
    ]
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["filename", "roll_number", "name", "application_number", "department", "year", "section"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for student in students_data:
            writer.writerow(student)
    
    print(f"✅ Created: {csv_path}")
    print(f"📊 Added {len(students_data)} students to CSV")

def create_json_metadata():
    """Create the JSON metadata file for compatibility"""
    print("\n📄 Creating JSON metadata file...")
    
    json_path = "data/student_dataset/students.json"
    
    # Read from CSV and convert to JSON format
    csv_path = "data/student_dataset/student_metadata.csv"
    students_json = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            student_json = {
                "roll_number": row["roll_number"],
                "name": row["name"],
                "application_number": row["application_number"],
                "department": row["department"],
                "year": row["year"],
                "section": row["section"],
                "photo_path": f"data/student_dataset/photos/{row['filename']}"
            }
            students_json.append(student_json)
    
    # Write JSON file
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(students_json, jsonfile, indent=4, ensure_ascii=False)
    
    print(f"✅ Created: {json_path}")

def create_photo_placeholders():
    """Create photo placeholder files"""
    print("\n📸 Creating photo placeholders...")
    
    photos_dir = Path("data/student_dataset/photos")
    photo_files = ["CS2021001.jpg", "CS2021002.jpg", "CS2021003.jpg"]
    
    for photo_file in photo_files:
        photo_path = photos_dir / photo_file
        if not photo_path.exists():
            # Create empty placeholder
            photo_path.touch()
            print(f"📁 Created placeholder: {photo_file}")
        else:
            print(f"✅ Already exists: {photo_file}")

def validate_setup():
    """Validate the setup"""
    print("\n🔍 VALIDATING SETUP...")
    
    required_files = [
        "data/student_dataset/student_metadata.csv",
        "data/student_dataset/students.json",
        "data/student_dataset/photos/CS2021001.jpg",
        "data/student_dataset/photos/CS2021002.jpg", 
        "data/student_dataset/photos/CS2021003.jpg"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 0:
                print(f"✅ {file_path}")
            else:
                print(f"⚠️  {file_path} (placeholder - replace with actual photo)")
        else:
            print(f"❌ {file_path} (missing)")
            all_good = False
    
    return all_good

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("🎓 COLLEGE DATASET SETUP COMPLETE!")
    print("="*70)
    
    print("\n📋 WHAT WAS CREATED:")
    print("✅ Directory structure: data/student_dataset/")
    print("✅ CSV metadata: student_metadata.csv (3 students)")
    print("✅ JSON metadata: students.json (for system compatibility)")
    print("✅ Photo placeholders: CS2021001.jpg, CS2021002.jpg, CS2021003.jpg")
    
    print("\n🔄 NEXT STEPS:")
    print("1. 📸 REPLACE PHOTO PLACEHOLDERS:")
    print("   - Copy actual student photos to data/student_dataset/photos/")
    print("   - Name them: CS2021001.jpg, CS2021002.jpg, CS2021003.jpg")
    print("   - Ensure photos are clear, well-lit, front-facing")
    
    print("\n2. 📊 TO ADD MORE STUDENTS:")
    print("   - Add rows to data/student_dataset/student_metadata.csv")
    print("   - Add corresponding photos to photos/ folder")
    print("   - Run: python setup_college_dataset.py (after installing face_recognition)")
    
    print("\n3. 🚀 INSTALL FACE RECOGNITION (Required for full functionality):")
    print("   pip install face-recognition dlib")
    
    print("\n4. 🧪 TEST THE SYSTEM:")
    print("   - With photos: python test_attendance_system.py")
    print("   - Full system: python src/main.py")
    
    print("\n📈 SCALING UP:")
    print("   - Current: 3 students (sample)")
    print("   - Expandable to: 1000+ students")
    print("   - Method: Add to CSV + photos, run bulk import")
    
    print("\n💡 SAMPLE CSV FORMAT FOR EXPANSION:")
    print("filename,roll_number,name,application_number,department,year,section")
    print("CS2021004.jpg,CS2021004,New Student,APP2021004,Computer Science,2021,B")

def main():
    """Main setup function"""
    print("🎓 COLLEGE DATASET SETUP - METHOD 3 (BULK IMPORT)")
    print("Basic Setup (No Face Recognition Dependencies)")
    print("="*70)
    
    try:
        # Step 1: Create directory structure
        create_directory_structure()
        
        # Step 2: Create CSV metadata
        create_csv_metadata()
        
        # Step 3: Create JSON metadata for compatibility
        create_json_metadata()
        
        # Step 4: Create photo placeholders
        create_photo_placeholders()
        
        # Step 5: Validate setup
        if validate_setup():
            print("\n✅ SETUP VALIDATION PASSED!")
        else:
            print("\n⚠️  SETUP VALIDATION HAD ISSUES")
        
        # Step 6: Print next steps
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
