#!/usr/bin/env python3
"""
Simple Camera Test and Fix
"""

import cv2
import time

def test_camera_simple():
    """Simple camera test"""
    print("🎥 Simple Camera Test")
    print("=" * 30)
    
    # Test different camera indices
    for i in range(5):
        print(f"\n🔍 Testing camera index {i}...")
        
        try:
            # Try different backends
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        
                        if ret and frame is not None:
                            print(f"✅ Camera {i} WORKING! ({frame.shape[1]}x{frame.shape[0]})")
                            
                            # Show a test frame
                            cv2.putText(frame, f"Camera {i} Working!", (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow(f'Camera {i} Test', frame)
                            cv2.waitKey(2000)  # Show for 2 seconds
                            cv2.destroyAllWindows()
                            
                            cap.release()
                            return i  # Return working camera index
                        else:
                            print(f"⚠️ Camera {i} opens but no frame")
                    else:
                        print(f"❌ Camera {i} cannot open")
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"❌ Camera {i} error: {e}")
                    
        except Exception as e:
            print(f"❌ Error testing camera {i}: {e}")
    
    print("\n❌ No working cameras found")
    return None

def fix_camera_issues():
    """Try to fix common camera issues"""
    print("\n🔧 Attempting to fix camera issues...")
    
    # Kill common camera processes
    import subprocess
    
    processes_to_kill = [
        "Teams.exe",
        "Zoom.exe", 
        "chrome.exe",
        "firefox.exe"
    ]
    
    for process in processes_to_kill:
        try:
            subprocess.run(['taskkill', '/f', '/im', process], 
                         capture_output=True, shell=True)
            print(f"🔧 Attempted to close {process}")
        except:
            pass
    
    print("⏳ Waiting 3 seconds for processes to close...")
    time.sleep(3)

if __name__ == "__main__":
    # First try to fix issues
    fix_camera_issues()
    
    # Then test cameras
    working_camera = test_camera_simple()
    
    if working_camera is not None:
        print(f"\n🎉 SUCCESS! Camera {working_camera} is working")
        print(f"💡 Update your main.py to use camera index {working_camera}")
        
        # Create a simple fix for main.py
        print("\n🔧 Creating camera fix...")
        
        fix_code = f"""
# Add this to the top of your main.py camera initialization:
WORKING_CAMERA_INDEX = {working_camera}

# Replace the camera initialization with:
self.cap = cv2.VideoCapture(WORKING_CAMERA_INDEX, cv2.CAP_DSHOW)
"""
        
        with open("camera_fix.txt", "w") as f:
            f.write(fix_code)
        
        print("✅ Camera fix saved to camera_fix.txt")
        
    else:
        print("\n❌ No cameras working. Try these solutions:")
        print("1. 🔄 Restart your computer")
        print("2. 🔌 Check camera connections")
        print("3. 🛠️ Update camera drivers")
        print("4. 🔒 Check Windows camera privacy settings")
        print("5. 🎥 Try a different camera")
