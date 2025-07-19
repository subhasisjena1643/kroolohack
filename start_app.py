#!/usr/bin/env python3
"""
Simple launcher for Classroom Engagement Analyzer
Handles all import path issues automatically
"""

import os
import sys
import subprocess

def main():
    """Launch the engagement analyzer"""
    print("🚀 Starting Classroom Engagement Analyzer...")
    print("📊 Industry-grade precision with continuous learning")
    print("🌐 Feedback interface will be available at: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the project directory
    os.chdir(current_dir)
    
    # Add current directory to Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = current_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    # Try to run the application
    try:
        # Run the main application
        result = subprocess.run([
            sys.executable, 'src/main.py'
        ], env=env, cwd=current_dir)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⏹️ Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
