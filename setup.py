#!/usr/bin/env python3
"""
Setup script for Real-time Classroom Engagement Analyzer
Hackathon Project by Subhasis & Sachin
"""

import subprocess
import sys
import os

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("🔧 Creating virtual environment...")
    
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine activation script path based on OS
    if os.name == 'nt':  # Windows
        activate_script = os.path.join("venv", "Scripts", "activate.bat")
        pip_path = os.path.join("venv", "Scripts", "pip.exe")
    else:  # Unix/Linux/MacOS
        activate_script = os.path.join("venv", "bin", "activate")
        pip_path = os.path.join("venv", "bin", "pip")
    
    print(f"✅ Virtual environment created!")
    print(f"📝 To activate: {activate_script}")
    
    return pip_path

def install_requirements(pip_path):
    """Install required packages"""
    print("📦 Installing required packages...")
    
    # Upgrade pip first
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    
    print("✅ All packages installed successfully!")

def create_project_structure():
    """Create project directory structure"""
    print("📁 Creating project structure...")
    
    directories = [
        "src",
        "src/modules",
        "src/utils",
        "config",
        "data",
        "data/models",
        "data/temp",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Project structure created!")

def main():
    """Main setup function"""
    print("🚀 Setting up Real-time Classroom Engagement Analyzer")
    print("=" * 50)
    
    try:
        # Create project structure
        create_project_structure()
        
        # Create virtual environment
        pip_path = create_virtual_environment()
        
        # Install requirements
        install_requirements(pip_path)
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate virtual environment")
        if os.name == 'nt':
            print("   Windows: venv\\Scripts\\activate")
        else:
            print("   Unix/Linux/MacOS: source venv/bin/activate")
        print("2. Run: python src/main.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
