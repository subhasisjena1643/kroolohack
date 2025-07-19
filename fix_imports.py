#!/usr/bin/env python3
"""
Fix import issues in the project
"""

import os
import re

def fix_imports_in_file(file_path):
    """Fix import statements in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix src.utils imports
        content = re.sub(r'from src\.utils\.', 'from utils.', content)
        content = re.sub(r'import src\.utils\.', 'import utils.', content)
        
        # Fix src.modules imports
        content = re.sub(r'from src\.modules\.', 'from modules.', content)
        content = re.sub(r'import src\.modules\.', 'import modules.', content)
        
        # Add missing typing imports
        if 'List[' in content and 'from typing import' in content:
            # Check if List is already imported
            if not re.search(r'from typing import.*List', content):
                content = re.sub(
                    r'from typing import ([^\\n]+)',
                    lambda m: f"from typing import {m.group(1)}, List" if 'List' not in m.group(1) else m.group(0),
                    content
                )
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in {file_path}")
            return True
        else:
            print(f"⏭️ No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix imports in all Python files"""
    print("🔧 Fixing import issues in the project...")
    
    # Directories to scan
    directories = ['src', 'config']
    
    fixed_count = 0
    
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if fix_imports_in_file(file_path):
                            fixed_count += 1
    
    print(f"\\n🎉 Fixed imports in {fixed_count} files")
    print("\\n🚀 Now try running: python run_app.py")

if __name__ == "__main__":
    main()
