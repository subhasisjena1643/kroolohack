#!/usr/bin/env python3
"""
Camera Issue Diagnostic and Fix Tool
Diagnoses and fixes common camera access issues
"""

import cv2
import numpy as np
import subprocess
import sys
import os
import time
from typing import List, Tuple, Optional

def check_camera_permissions():
    """Check if camera permissions are enabled"""
    print("🔍 Checking camera permissions...")
    
    try:
        # Try to access camera with different backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_MSMF, "Media Foundation (Windows)"),
            (cv2.CAP_V4L2, "Video4Linux (Linux)"),
            (cv2.CAP_ANY, "Any Available")
        ]
        
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print(f"✅ Camera accessible with {name}")
                        return True, backend
                    else:
                        print(f"⚠️ Camera opens but no frame with {name}")
                else:
                    print(f"❌ Camera not accessible with {name}")
            except Exception as e:
                print(f"❌ Error with {name}: {e}")
        
        return False, None
        
    except Exception as e:
        print(f"❌ Permission check failed: {e}")
        return False, None

def find_available_cameras() -> List[int]:
    """Find all available camera indices"""
    print("🔍 Scanning for available cameras...")
    
    available_cameras = []
    
    # Test camera indices 0-10
    for i in range(11):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"✅ Camera {i}: Working ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    print(f"⚠️ Camera {i}: Opens but no frame")
                cap.release()
            else:
                print(f"❌ Camera {i}: Not accessible")
        except Exception as e:
            print(f"❌ Camera {i}: Error - {e}")
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    return available_cameras

def kill_camera_processes():
    """Kill processes that might be using the camera"""
    print("🔧 Checking for processes using camera...")
    
    # Common processes that use camera
    camera_processes = [
        "Teams.exe",
        "Zoom.exe", 
        "Skype.exe",
        "chrome.exe",
        "firefox.exe",
        "obs64.exe",
        "obs32.exe",
        "CameraApp.exe",
        "WindowsCamera.exe"
    ]
    
    killed_processes = []
    
    try:
        # Get list of running processes
        result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
        running_processes = result.stdout.lower()
        
        for process in camera_processes:
            if process.lower() in running_processes:
                try:
                    subprocess.run(['taskkill', '/f', '/im', process], 
                                 capture_output=True, shell=True)
                    killed_processes.append(process)
                    print(f"🔧 Killed process: {process}")
                except Exception as e:
                    print(f"⚠️ Could not kill {process}: {e}")
        
        if killed_processes:
            print(f"✅ Killed {len(killed_processes)} camera processes")
            time.sleep(2)  # Wait for processes to fully close
        else:
            print("ℹ️ No camera processes found to kill")
            
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
    
    return killed_processes

def test_camera_with_settings(camera_index: int) -> bool:
    """Test camera with different settings"""
    print(f"🧪 Testing camera {camera_index} with different settings...")
    
    settings_to_try = [
        # (width, height, fps)
        (640, 480, 30),
        (1280, 720, 30),
        (800, 600, 30),
        (320, 240, 30),
        (640, 480, 15),
    ]
    
    for width, height, fps in settings_to_try:
        try:
            cap = cv2.VideoCapture(camera_index)
            
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    print(f"✅ Working: {actual_width}x{actual_height} @ {actual_fps}fps")
                    cap.release()
                    return True
                else:
                    print(f"❌ No frame at {width}x{height}")
            else:
                print(f"❌ Cannot open at {width}x{height}")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Error testing {width}x{height}: {e}")
    
    return False

def create_camera_test_window(camera_index: int):
    """Create a test window to verify camera is working"""
    print(f"📹 Opening test window for camera {camera_index}...")
    print("Press 'q' to close the test window")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return False
        
        # Set reasonable defaults
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add frame counter and instructions
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Camera is working!", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f'Camera {camera_index} Test', frame)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Camera {camera_index} test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def main():
    """Main diagnostic and fix function"""
    print("🎥 CAMERA DIAGNOSTIC AND FIX TOOL")
    print("=" * 50)
    
    # Step 1: Kill camera processes
    print("\n📋 STEP 1: Freeing camera resources...")
    killed_processes = kill_camera_processes()
    
    # Step 2: Check permissions
    print("\n📋 STEP 2: Checking camera permissions...")
    has_permission, working_backend = check_camera_permissions()
    
    # Step 3: Find available cameras
    print("\n📋 STEP 3: Scanning for cameras...")
    available_cameras = find_available_cameras()
    
    # Step 4: Test cameras
    if available_cameras:
        print(f"\n📋 STEP 4: Testing {len(available_cameras)} available cameras...")
        
        working_cameras = []
        for camera_idx in available_cameras:
            if test_camera_with_settings(camera_idx):
                working_cameras.append(camera_idx)
        
        if working_cameras:
            print(f"\n✅ SUCCESS: {len(working_cameras)} working cameras found!")
            print(f"Working cameras: {working_cameras}")
            
            # Test the first working camera
            test_camera = working_cameras[0]
            print(f"\n📹 Testing camera {test_camera} with live preview...")
            
            user_input = input(f"Open test window for camera {test_camera}? (y/n): ")
            if user_input.lower() == 'y':
                create_camera_test_window(test_camera)
            
            # Update main.py to use working camera
            print(f"\n🔧 RECOMMENDED: Update your camera index to {test_camera}")
            
            return True
        else:
            print("\n❌ No working cameras found")
    else:
        print("\n❌ No cameras detected")
    
    # Step 5: Provide solutions
    print("\n📋 STEP 5: Troubleshooting recommendations...")
    print("\n🔧 POSSIBLE SOLUTIONS:")
    print("1. 🔄 Restart your computer")
    print("2. 🔌 Check camera cable connections")
    print("3. 🛠️ Update camera drivers")
    print("4. 🔒 Check Windows camera privacy settings")
    print("5. 🎥 Try external USB camera")
    print("6. 💻 Check Device Manager for camera issues")
    
    return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Camera issues resolved!")
        else:
            print("\n⚠️ Manual intervention may be required")
    except KeyboardInterrupt:
        print("\n\n⏹️ Diagnostic cancelled by user")
    except Exception as e:
        print(f"\n❌ Diagnostic tool error: {e}")
