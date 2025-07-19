#!/usr/bin/env python3
"""
Simple Virtual Environment Setup
Step-by-step setup with clear error handling
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_step(step_num, description):
    """Print step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def run_cmd(command, description=""):
    """Run command with error handling"""
    print(f"Running: {description}")
    print(f"   Command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"   SUCCESS!")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"   FAILED (exit code: {result.returncode})")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"   EXCEPTION: {e}")
        return False

def main():
    """Main setup function"""
    print("VIRTUAL ENVIRONMENT SETUP - SIMPLE VERSION")
    print("=" * 60)
    
    # Step 1: Check Python
    print_step(1, "Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    
    print("Python version OK")
    
    # Step 2: Create virtual environment
    print_step(2, "Creating Virtual Environment")
    
    if Path("venv").exists():
        print("Virtual environment already exists")
    else:
        success = run_cmd(f"{sys.executable} -m venv venv", "Creating venv")
        if not success:
            print("FAILED to create virtual environment")
            return False
    
    # Step 3: Determine commands
    print_step(3, "Setting up commands")
    
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
        activate_cmd = "venv\\Scripts\\activate"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
        activate_cmd = "source venv/bin/activate"
    
    print(f"Pip command: {pip_cmd}")
    print(f"Python command: {python_cmd}")
    print(f"Activate command: {activate_cmd}")
    
    # Step 4: Upgrade pip
    print_step(4, "Upgrading pip")
    success = run_cmd(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    if not success:
        print("WARNING: Pip upgrade failed, continuing...")
    
    # Step 5: Install basic packages
    print_step(5, "Installing Basic Packages")
    
    basic_packages = [
        "wheel",
        "setuptools",
        "numpy==1.24.3",
        "pillow==10.0.1"
    ]
    
    for package in basic_packages:
        success = run_cmd(f"{pip_cmd} install {package}", f"Installing {package}")
        if not success:
            print(f"WARNING: Failed to install {package}, continuing...")
    
    # Step 6: Install MediaPipe
    print_step(6, "Installing MediaPipe")
    
    mediapipe_versions = [
        "mediapipe==0.10.7",
        "mediapipe==0.10.3", 
        "mediapipe==0.9.3.0"
    ]
    
    mediapipe_success = False
    for version in mediapipe_versions:
        print(f"\nTrying {version}...")
        success = run_cmd(f"{pip_cmd} install {version}", f"Installing {version}")

        if success:
            # Test import
            test_success = run_cmd(
                f'{python_cmd} -c "import mediapipe; print(\'MediaPipe imported successfully\')"',
                "Testing MediaPipe import"
            )

            if test_success:
                print(f"SUCCESS: {version} installed and working!")
                mediapipe_success = True
                break
            else:
                print(f"WARNING: {version} installed but import failed")
        else:
            print(f"FAILED: {version} installation failed")

    if not mediapipe_success:
        print("FAILED: All MediaPipe versions failed")
        print("TIP: Try installing Visual C++ Redistributable and run again")
    
    # Step 7: Install remaining requirements
    print_step(7, "Installing Remaining Requirements")
    
    if Path("requirements.txt").exists():
        success = run_cmd(f"{pip_cmd} install -r requirements.txt", "Installing requirements.txt")
        if not success:
            print("WARNING: Some packages in requirements.txt failed")
    else:
        print("WARNING: requirements.txt not found")
    
    # Step 8: Create activation scripts
    print_step(8, "Creating Activation Scripts")
    
    if platform.system() == "Windows":
        # Create batch file (without emojis for Windows compatibility)
        batch_content = f'''@echo off
echo Activating Virtual Environment...
call {activate_cmd}
echo Virtual environment activated!
echo Run: python src/main.py
cmd /k
'''
        with open("activate.bat", "w", encoding='utf-8') as f:
            f.write(batch_content)
        print("Created activate.bat")
    else:
        # Create shell script
        shell_content = f'''#!/bin/bash
echo "Activating Virtual Environment..."
{activate_cmd}
echo "Virtual environment activated!"
echo "Run: python src/main.py"
exec "$SHELL"
'''
        with open("activate.sh", "w", encoding='utf-8') as f:
            f.write(shell_content)
        os.chmod("activate.sh", 0o755)
        print("Created activate.sh")
    
    # Step 9: Create test script
    print_step(9, "Creating Test Script")
    
    test_content = '''#!/usr/bin/env python3
"""Test all packages"""

def test_imports():
    packages = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("flask", "Flask"),
        ("pandas", "Pandas")
    ]

    print("Testing package imports...")
    success_count = 0

    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")

    print(f"\\nResult: {success_count}/{len(packages)} packages working")
    return success_count == len(packages)

if __name__ == "__main__":
    test_imports()
'''

    with open("test_packages.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    print("Created test_packages.py")
    
    # Final summary
    print_step("FINAL", "Setup Complete!")
    
    print("NEXT STEPS:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   • Double-click activate.bat, OR")
        print(f"   • Run: {activate_cmd}")
    else:
        print(f"   • Run: ./activate.sh, OR")
        print(f"   • Run: {activate_cmd}")
    
    print("\\n2. Test packages:")
    print("   python test_packages.py")
    
    print("\\n3. If MediaPipe failed:")
    print("   • Install Visual C++ Redistributable")
    print("   • Try: pip install mediapipe --no-cache-dir")
    
    print("\\n4. Setup continuous learning:")
    print("   python setup_continuous_learning.py")
    
    print("\\n5. Start application:")
    print("   python src/main.py")
    
    if mediapipe_success:
        print("\\nSUCCESS: Virtual environment ready with MediaPipe!")
    else:
        print("\\nPARTIAL SUCCESS: Virtual environment ready, MediaPipe needs attention")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n❌ Setup interrupted by user")
    except Exception as e:
        print(f"\\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
