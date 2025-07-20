#!/usr/bin/env python3
"""
Launcher script for Classroom Engagement Analyzer
Handles import path issues
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Now import and run the main application
try:
    from src.main import EngagementAnalyzer
    
    if __name__ == "__main__":
        print("🚀 Starting Classroom Engagement Analyzer...")
        print("📊 Industry-grade precision with continuous learning")
        print("🌐 Feedback interface will be available at: http://localhost:5001")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        app = EngagementAnalyzer()
        app.start()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure all required packages are installed:")
    print("   pip install -r requirements_minimal.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
