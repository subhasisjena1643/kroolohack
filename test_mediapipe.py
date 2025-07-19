#!/usr/bin/env python3
"""
Test MediaPipe Installation
Quick test to verify MediaPipe is working correctly
"""

def test_mediapipe_import():
    """Test basic MediaPipe import"""
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully!")
        print(f"📦 MediaPipe version: {mp.__version__}")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False

def test_mediapipe_solutions():
    """Test MediaPipe solutions"""
    try:
        import mediapipe as mp
        
        # Test Face Detection
        face_detection = mp.solutions.face_detection
        print("✅ Face Detection solution loaded")
        
        # Test Face Mesh
        face_mesh = mp.solutions.face_mesh
        print("✅ Face Mesh solution loaded")
        
        # Test Pose
        pose = mp.solutions.pose
        print("✅ Pose solution loaded")
        
        # Test Hands
        hands = mp.solutions.hands
        print("✅ Hands solution loaded")
        
        # Test Holistic
        holistic = mp.solutions.holistic
        print("✅ Holistic solution loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe solutions test failed: {e}")
        return False

def test_mediapipe_initialization():
    """Test MediaPipe model initialization"""
    try:
        import mediapipe as mp
        
        # Test Face Detection initialization
        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            print("✅ Face Detection model initialized")
        
        # Test Pose initialization
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
            print("✅ Pose model initialized")
        
        # Test Hands initialization
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
            print("✅ Hands model initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe initialization test failed: {e}")
        return False

def main():
    """Run all MediaPipe tests"""
    print("🔧 Testing MediaPipe Installation...")
    print("=" * 50)
    
    # Test 1: Basic import
    print("\n📦 Test 1: Basic Import")
    import_success = test_mediapipe_import()
    
    if not import_success:
        print("\n❌ MediaPipe import failed. Please fix installation first.")
        print("\n🔧 Try these solutions:")
        print("1. pip uninstall mediapipe")
        print("2. pip install mediapipe==0.10.7")
        print("3. Install Visual C++ Redistributable")
        print("4. Try: conda install -c conda-forge mediapipe")
        return False
    
    # Test 2: Solutions loading
    print("\n🧩 Test 2: Solutions Loading")
    solutions_success = test_mediapipe_solutions()
    
    if not solutions_success:
        print("\n❌ MediaPipe solutions loading failed.")
        return False
    
    # Test 3: Model initialization
    print("\n🤖 Test 3: Model Initialization")
    init_success = test_mediapipe_initialization()
    
    if not init_success:
        print("\n❌ MediaPipe model initialization failed.")
        return False
    
    # All tests passed
    print("\n" + "=" * 50)
    print("🎉 ALL MEDIAPIPE TESTS PASSED!")
    print("✅ MediaPipe is ready for industry-grade analysis")
    print("🚀 You can now run: python setup_continuous_learning.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main()
