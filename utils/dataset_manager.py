"""
Student Dataset Management Utility
Tools for managing student photos, face encodings, and database
"""

import os
import json
import pickle
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
import sqlite3
from datetime import datetime
import shutil

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded in dataset manager")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not available in dataset manager")

class StudentDatasetManager:
    """Utility class for managing student dataset"""
    
    def __init__(self, dataset_path: str = "data/student_dataset"):
        self.dataset_path = dataset_path
        self.photos_path = os.path.join(dataset_path, "photos")
        self.students_file = os.path.join(dataset_path, "students.json")
        self.encodings_file = os.path.join(dataset_path, "face_encodings.pkl")
        
        # Create directories
        os.makedirs(self.photos_path, exist_ok=True)
        os.makedirs(dataset_path, exist_ok=True)
    
    def add_student(self, roll_number: str, name: str, application_number: str,
                   photo_path: str, department: str = "", year: str = "", section: str = "") -> bool:
        """Add a new student to the dataset"""
        try:
            # Load existing students
            students = self.load_students()
            
            # Check if student already exists
            for student in students:
                if student['roll_number'] == roll_number:
                    print(f"Student {roll_number} already exists. Updating...")
                    students.remove(student)
                    break
            
            # Copy photo to dataset
            photo_filename = f"{roll_number}.jpg"
            target_photo_path = os.path.join(self.photos_path, photo_filename)
            
            if os.path.exists(photo_path):
                shutil.copy2(photo_path, target_photo_path)
                print(f"Photo copied to {target_photo_path}")
            else:
                print(f"Warning: Photo not found at {photo_path}")
                return False
            
            # Add student data
            student_data = {
                "roll_number": roll_number,
                "name": name,
                "application_number": application_number,
                "department": department,
                "year": year,
                "section": section,
                "photo_path": target_photo_path
            }
            
            students.append(student_data)
            
            # Save students data
            with open(self.students_file, 'w') as f:
                json.dump(students, f, indent=4)
            
            # Regenerate face encodings
            self.generate_face_encodings()
            
            print(f"Successfully added student: {name} ({roll_number})")
            return True
            
        except Exception as e:
            print(f"Error adding student: {e}")
            return False
    
    def load_students(self) -> List[Dict]:
        """Load students from JSON file"""
        if os.path.exists(self.students_file):
            with open(self.students_file, 'r') as f:
                return json.load(f)
        return []
    
    def generate_face_encodings(self) -> bool:
        """Generate face encodings for all students using DeepFace"""
        try:
            if not DEEPFACE_AVAILABLE:
                print("❌ DeepFace not available. Cannot generate encodings.")
                return False

            students = self.load_students()
            encodings = {}

            print("Generating face encodings using DeepFace...")

            for student in students:
                roll_number = student['roll_number']
                photo_path = student['photo_path']
                name = student['name']

                if os.path.exists(photo_path):
                    # Check if file is not empty
                    if os.path.getsize(photo_path) == 0:
                        print(f"⚠️  Empty photo file for {name} ({roll_number})")
                        continue

                    try:
                        # INDUSTRIAL-GRADE MULTI-MODEL EMBEDDING GENERATION
                        print(f"🧠 Generating INDUSTRIAL-GRADE encoding for {name}...")

                        # Preprocess image for better recognition
                        img = cv2.imread(photo_path)
                        if img is None:
                            print(f"❌ Could not load image for {name}")
                            continue

                        # Apply lighting normalization
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        img_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                        # Save preprocessed image temporarily
                        temp_path = photo_path.replace('.jpg', '_temp.jpg').replace('.png', '_temp.png')
                        cv2.imwrite(temp_path, img_normalized)

                        # Use Facenet with enhanced preprocessing for maximum robustness
                        try:
                            embedding_objs = DeepFace.represent(
                                img_path=temp_path,
                                model_name='Facenet',  # Most robust and compatible model
                                enforce_detection=True,
                                detector_backend='opencv'
                            )

                            if embedding_objs and len(embedding_objs) > 0:
                                embedding = np.array(embedding_objs[0]['embedding'])
                                print(f"  ✅ Facenet (Enhanced): {len(embedding)} dimensions")

                                # Store the robust encoding
                                encodings[roll_number] = embedding
                                print(f"✅ INDUSTRIAL-GRADE ENCODING for {name} ({roll_number}) - Enhanced Facenet")
                            else:
                                print(f"  ❌ Facenet: No face detected for {name}")

                        except Exception as model_error:
                            print(f"  ❌ Facenet failed for {name}: {model_error}")

                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    except Exception as deepface_error:
                        print(f"❌ DeepFace failed for {name} ({roll_number}): {deepface_error}")

                else:
                    print(f"❌ Photo not found for {name} ({roll_number}): {photo_path}")

            # Save encodings
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(encodings, f)

            print(f"✅ DeepFace encodings saved for {len(encodings)} students")
            return True

        except Exception as e:
            print(f"❌ Error generating face encodings: {e}")
            return False
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the dataset and return statistics"""
        students = self.load_students()
        stats = {
            'total_students': len(students),
            'valid_photos': 0,
            'missing_photos': 0,
            'valid_encodings': 0,
            'missing_encodings': 0,
            'issues': []
        }
        
        # Load encodings if available
        encodings = {}
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                encodings = pickle.load(f)
        
        for student in students:
            roll_number = student['roll_number']
            photo_path = student['photo_path']
            name = student['name']
            
            # Check photo
            if os.path.exists(photo_path):
                stats['valid_photos'] += 1
            else:
                stats['missing_photos'] += 1
                stats['issues'].append(f"Missing photo for {name} ({roll_number})")
            
            # Check encoding
            if roll_number in encodings:
                stats['valid_encodings'] += 1
            else:
                stats['missing_encodings'] += 1
                stats['issues'].append(f"Missing encoding for {name} ({roll_number})")
        
        return stats
    
    def remove_student(self, roll_number: str) -> bool:
        """Remove a student from the dataset"""
        try:
            students = self.load_students()
            
            # Find and remove student
            for student in students:
                if student['roll_number'] == roll_number:
                    # Remove photo
                    photo_path = student['photo_path']
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                    
                    # Remove from list
                    students.remove(student)
                    
                    # Save updated list
                    with open(self.students_file, 'w') as f:
                        json.dump(students, f, indent=4)
                    
                    # Regenerate encodings
                    self.generate_face_encodings()
                    
                    print(f"Removed student: {student['name']} ({roll_number})")
                    return True
            
            print(f"Student {roll_number} not found")
            return False
            
        except Exception as e:
            print(f"Error removing student: {e}")
            return False
    
    def bulk_import_from_folder(self, folder_path: str, csv_file: str = None) -> int:
        """Bulk import students from a folder of photos with CSV metadata"""
        try:
            imported_count = 0
            skipped_count = 0
            error_count = 0

            print(f"Starting bulk import from: {folder_path}")
            if csv_file:
                print(f"Using metadata from: {csv_file}")

            # Load metadata from CSV if provided
            metadata = {}
            if csv_file and os.path.exists(csv_file):
                import csv
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get('filename', '').strip()
                        if filename:
                            metadata[filename] = {
                                'roll_number': row.get('roll_number', '').strip(),
                                'name': row.get('name', '').strip(),
                                'application_number': row.get('application_number', '').strip(),
                                'department': row.get('department', '').strip(),
                                'year': row.get('year', '').strip(),
                                'section': row.get('section', '').strip()
                            }
                print(f"Loaded metadata for {len(metadata)} students")
            else:
                print("No CSV metadata file provided or file not found")

            # Process photos based on CSV metadata
            if metadata:
                # Process files listed in CSV
                for filename, meta in metadata.items():
                    photo_path = os.path.join(folder_path, filename)

                    if not os.path.exists(photo_path):
                        print(f"⚠️  Photo not found: {filename}")
                        skipped_count += 1
                        continue

                    # Validate required fields
                    roll_number = meta.get('roll_number')
                    name = meta.get('name')
                    application_number = meta.get('application_number')

                    if not all([roll_number, name, application_number]):
                        print(f"⚠️  Missing required data for {filename}")
                        skipped_count += 1
                        continue

                    # Add student
                    try:
                        if self.add_student(
                            roll_number=roll_number,
                            name=name,
                            application_number=application_number,
                            photo_path=photo_path,
                            department=meta.get('department', ''),
                            year=meta.get('year', ''),
                            section=meta.get('section', '')
                        ):
                            print(f"✅ Added: {name} ({roll_number})")
                            imported_count += 1
                        else:
                            print(f"❌ Failed to add: {name} ({roll_number})")
                            error_count += 1
                    except Exception as e:
                        print(f"❌ Error adding {name}: {e}")
                        error_count += 1
            else:
                # Fallback: Process all image files in folder
                print("Processing all image files in folder...")
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Extract roll number from filename
                        roll_number = os.path.splitext(filename)[0]
                        photo_path = os.path.join(folder_path, filename)

                        try:
                            if self.add_student(
                                roll_number=roll_number,
                                name=f'Student {roll_number}',
                                application_number=f'APP{roll_number}',
                                photo_path=photo_path
                            ):
                                print(f"✅ Added: Student {roll_number}")
                                imported_count += 1
                            else:
                                error_count += 1
                        except Exception as e:
                            print(f"❌ Error adding {roll_number}: {e}")
                            error_count += 1

            # Summary
            print(f"\n📊 BULK IMPORT SUMMARY:")
            print(f"✅ Successfully imported: {imported_count} students")
            print(f"⚠️  Skipped: {skipped_count} students")
            print(f"❌ Errors: {error_count} students")
            print(f"📁 Total processed: {imported_count + skipped_count + error_count}")

            return imported_count

        except Exception as e:
            print(f"❌ Critical error in bulk import: {e}")
            return 0
    
    def export_attendance_report(self, db_path: str, output_file: str = None) -> str:
        """Export attendance report from database"""
        try:
            if output_file is None:
                output_file = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT roll_number, name, date, entry_time, exit_time, total_duration,
                       engagement_score, participation_score, attention_score
                FROM attendance
                ORDER BY date DESC, entry_time DESC
            ''')
            
            rows = cursor.fetchall()
            
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Roll Number', 'Name', 'Date', 'Entry Time', 'Exit Time', 
                               'Duration (min)', 'Engagement Score', 'Participation Score', 'Attention Score'])
                writer.writerows(rows)
            
            conn.close()
            print(f"Attendance report exported to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error exporting attendance report: {e}")
            return ""

def main():
    """Command line interface for dataset management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Student Dataset Manager')
    parser.add_argument('--add', action='store_true', help='Add a new student')
    parser.add_argument('--roll', type=str, help='Roll number')
    parser.add_argument('--name', type=str, help='Student name')
    parser.add_argument('--app-num', type=str, help='Application number')
    parser.add_argument('--photo', type=str, help='Photo path')
    parser.add_argument('--dept', type=str, default='', help='Department')
    parser.add_argument('--year', type=str, default='', help='Year')
    parser.add_argument('--section', type=str, default='', help='Section')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    parser.add_argument('--generate-encodings', action='store_true', help='Generate face encodings')
    parser.add_argument('--bulk-import', type=str, help='Bulk import from folder')
    parser.add_argument('--csv-metadata', type=str, help='CSV file with metadata for bulk import')
    
    args = parser.parse_args()
    
    manager = StudentDatasetManager()
    
    if args.add:
        if not all([args.roll, args.name, args.app_num, args.photo]):
            print("Error: --roll, --name, --app-num, and --photo are required for adding a student")
            return
        
        manager.add_student(args.roll, args.name, args.app_num, args.photo, 
                          args.dept, args.year, args.section)
    
    elif args.validate:
        stats = manager.validate_dataset()
        print("\nDataset Validation Results:")
        print(f"Total Students: {stats['total_students']}")
        print(f"Valid Photos: {stats['valid_photos']}")
        print(f"Missing Photos: {stats['missing_photos']}")
        print(f"Valid Encodings: {stats['valid_encodings']}")
        print(f"Missing Encodings: {stats['missing_encodings']}")
        
        if stats['issues']:
            print("\nIssues Found:")
            for issue in stats['issues']:
                print(f"  - {issue}")
    
    elif args.generate_encodings:
        manager.generate_face_encodings()
    
    elif args.bulk_import:
        manager.bulk_import_from_folder(args.bulk_import, args.csv_metadata)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
