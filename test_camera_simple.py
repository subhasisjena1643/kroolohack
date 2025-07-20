#!/usr/bin/env python3
"""
Simple camera test to check if camera is accessible
"""

import cv2
import numpy as np

def test_camera():
    """Test camera access"""
    print("🎥 Testing camera access...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"✅ Camera {camera_index} opened successfully!")
            
            # Test reading a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Successfully read frame from camera {camera_index}")
                print(f"   Frame shape: {frame.shape}")
                
                # Add test labels to the frame
                cv2.putText(frame, f"Camera {camera_index} - WORKING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw a test face rectangle with label
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 3)
                cv2.putText(frame, "TEST FACE LABEL", (105, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f'Camera {camera_index} Test', frame)
                print(f"📺 Showing camera {camera_index} feed. Press any key to continue...")
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                return camera_index
            else:
                print(f"❌ Could not read frame from camera {camera_index}")
        else:
            print(f"❌ Could not open camera {camera_index}")
        
        cap.release()
    
    print("❌ No working camera found!")
    return None

def main():
    """Main test function"""
    print("🧪 SIMPLE CAMERA TEST")
    print("="*40)
    
    working_camera = test_camera()
    
    if working_camera is not None:
        print(f"\n✅ CAMERA {working_camera} IS WORKING!")
        print("\n🔧 TO FIX THE MAIN SYSTEM:")
        print(f"1. Update camera index to {working_camera} in config")
        print("2. The drawing functions should work with this camera")
        print("3. Labels will appear once camera is properly initialized")
    else:
        print("\n❌ NO CAMERA AVAILABLE!")
        print("\n🔧 POSSIBLE SOLUTIONS:")
        print("1. Check if camera is being used by another application")
        print("2. Try running as administrator")
        print("3. Check camera permissions")
        print("4. Try external USB camera")

if __name__ == "__main__":
    main()
