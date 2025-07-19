#!/usr/bin/env python3
"""
Virtual Environment Setup Script
Creates venv and installs all dependencies with proper MediaPipe compatibility
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        
        if result.returncode == 0:
            print(f"   ✅ {description} completed successfully")
        else:
            print(f"   ⚠️ {description} completed with warnings")
            if result.stderr:
                print(f"   Warning: {result.stderr.strip()}")
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error in {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ Python 3 is required")
        return False
    
    if version.minor < 8:
        print("❌ Python 3.8+ is required for MediaPipe")
        return False
    
    if version.minor > 11:
        print("⚠️ Python 3.11+ may have MediaPipe compatibility issues")
        print("   Consider using Python 3.8-3.10 for best compatibility")
    
    print("✅ Python version is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    success = run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )
    
    if success:
        print("✅ Virtual environment created successfully")
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def get_pip_command():
    """Get the correct pip command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    else:
        return "venv/bin/pip"

def upgrade_pip():
    """Upgrade pip in virtual environment"""
    pip_cmd = get_pip_command()
    
    success = run_command(
        f"{pip_cmd} install --upgrade pip",
        "Upgrading pip"
    )
    
    return success

def install_requirements():
    """Install requirements with special handling for problematic packages"""
    pip_cmd = get_pip_command()
    
    # First, install basic dependencies
    print("\n📦 Installing basic dependencies...")
    basic_deps = [
        "numpy==1.24.3",
        "pillow==10.0.1",
        "opencv-python==4.8.1.78"
    ]
    
    for dep in basic_deps:
        success = run_command(
            f"{pip_cmd} install {dep}",
            f"Installing {dep}"
        )
        if not success:
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    # Install MediaPipe with special handling
    print("\n🎯 Installing MediaPipe...")
    mediapipe_success = install_mediapipe(pip_cmd)
    
    # Install remaining requirements
    print("\n📋 Installing remaining requirements...")
    success = run_command(
        f"{pip_cmd} install -r requirements.txt",
        "Installing requirements.txt",
        check=False  # Don't fail if some packages have issues
    )
    
    return mediapipe_success

def install_mediapipe(pip_cmd):
    """Install MediaPipe with fallback options"""
    
    # Try different MediaPipe installation strategies
    strategies = [
        ("mediapipe==0.10.7", "Installing MediaPipe 0.10.7"),
        ("mediapipe==0.10.3", "Installing MediaPipe 0.10.3 (fallback)"),
        ("mediapipe==0.9.3.0", "Installing MediaPipe 0.9.3.0 (stable fallback)"),
        ("mediapipe --no-cache-dir", "Installing MediaPipe (no cache)"),
    ]
    
    for strategy, description in strategies:
        print(f"\n🎯 {description}")
        success = run_command(
            f"{pip_cmd} install {strategy}",
            description,
            check=False
        )
        
        if success:
            # Test MediaPipe import
            test_success = test_mediapipe_import(pip_cmd)
            if test_success:
                print("✅ MediaPipe installed and tested successfully!")
                return True
            else:
                print("⚠️ MediaPipe installed but import test failed, trying next strategy...")
        else:
            print("⚠️ Installation failed, trying next strategy...")
    
    print("❌ All MediaPipe installation strategies failed")
    return False

def test_mediapipe_import(pip_cmd):
    """Test MediaPipe import in virtual environment"""
    python_cmd = get_pip_command().replace("pip", "python")

    test_command = f'{python_cmd} -c "import mediapipe as mp; print(\'MediaPipe\', mp.__version__, \'imported successfully\')"'

    success = run_command(
        test_command,
        "Testing MediaPipe import",
        check=False
    )

    return success

def create_activation_scripts():
    """Create convenient activation scripts"""
    
    if platform.system() == "Windows":
        # Windows batch file
        batch_content = '''@echo off
echo 🐍 Activating Python Virtual Environment...
call venv\\Scripts\\activate
echo ✅ Virtual environment activated!
echo 🚀 You can now run: python src/main.py
cmd /k
'''
        with open("activate_venv.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created activate_venv.bat")
        
        # PowerShell script
        ps_content = '''Write-Host "🐍 Activating Python Virtual Environment..." -ForegroundColor Green
& .\\venv\\Scripts\\Activate.ps1
Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host "🚀 You can now run: python src/main.py" -ForegroundColor Yellow
'''
        with open("activate_venv.ps1", "w") as f:
            f.write(ps_content)
        print("✅ Created activate_venv.ps1")
        
    else:
        # Unix shell script
        shell_content = '''#!/bin/bash
echo "🐍 Activating Python Virtual Environment..."
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "🚀 You can now run: python src/main.py"
exec "$SHELL"
'''
        with open("activate_venv.sh", "w") as f:
            f.write(shell_content)
        os.chmod("activate_venv.sh", 0o755)
        print("✅ Created activate_venv.sh")

def create_requirements_check():
    """Create a script to check all requirements"""
    check_content = '''#!/usr/bin/env python3
"""
Check all requirements are properly installed
"""

import sys
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed or not importable")
        return False

def main():
    """Check all critical packages"""
    print("🔍 Checking installed packages...")
    print("=" * 50)
    
    packages = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("flask", "flask"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("librosa", "librosa"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            success_count += 1
    
    print("=" * 50)
    print(f"📊 Results: {success_count}/{total_count} packages working")
    
    if success_count == total_count:
        print("🎉 All packages are working correctly!")
        return True
    else:
        print("⚠️ Some packages need attention")
        return False

if __name__ == "__main__":
    main()
'''
    
    with open("check_requirements.py", "w") as f:
        f.write(check_content)
    print("✅ Created check_requirements.py")

def print_next_steps():
    """Print next steps for the user"""
    activation_cmd = get_activation_command()
    
    print("\n" + "=" * 60)
    print("🎉 VIRTUAL ENVIRONMENT SETUP COMPLETE!")
    print("=" * 60)
    
    print(f"\n🐍 To activate the virtual environment:")
    if platform.system() == "Windows":
        print("   Option 1: Double-click activate_venv.bat")
        print("   Option 2: Run activate_venv.ps1 in PowerShell")
        print(f"   Option 3: Run {activation_cmd}")
    else:
        print(f"   Option 1: Run ./activate_venv.sh")
        print(f"   Option 2: Run {activation_cmd}")
    
    print(f"\n🔍 To check all packages are working:")
    print("   python check_requirements.py")
    
    print(f"\n🚀 To start the application:")
    print("   python setup_continuous_learning.py  # Setup datasets")
    print("   python src/main.py                   # Start application")
    
    print(f"\n📊 To test MediaPipe specifically:")
    print("   python test_mediapipe.py")
    
    print(f"\n💡 Tips:")
    print("   • Always activate the venv before running Python scripts")
    print("   • If MediaPipe fails, check Visual C++ Redistributables")
    print("   • Use check_requirements.py to diagnose issues")

def main():
    """Main setup function"""
    print("🔧 Setting up Virtual Environment for Engagement Analyzer")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    print(f"\n📁 Creating virtual environment...")
    if not create_virtual_environment():
        return False
    
    # Upgrade pip
    print(f"\n⬆️ Upgrading pip...")
    if not upgrade_pip():
        print("⚠️ Pip upgrade failed, continuing anyway...")
    
    # Install requirements
    print(f"\n📦 Installing requirements...")
    mediapipe_success = install_requirements()
    
    # Create helper scripts
    print(f"\n📝 Creating helper scripts...")
    create_activation_scripts()
    create_requirements_check()
    
    # Print results
    if mediapipe_success:
        print("\n✅ Setup completed successfully with MediaPipe!")
    else:
        print("\n⚠️ Setup completed but MediaPipe may need attention")
        print("   Try running check_requirements.py to diagnose issues")
    
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
