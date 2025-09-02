#!/usr/bin/env python3
"""
Setup script for LLM Study Analysis

This script helps set up the environment and validates the data file.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'outputs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually using:")
        print("   pip install -r requirements.txt")
        return False

def check_data_file():
    """Check if a CSV data file exists in the data directory."""
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    
    if csv_files:
        if len(csv_files) == 1:
            print(f"✅ Data file found: {csv_files[0].name}")
        else:
            print(f"✅ Data files found: {len(csv_files)} CSV files")
            print(f"   Will use: {csv_files[0].name}")
        return True
    else:
        print("⚠️  No CSV data file found.")
        print(f"   Please place your CSV file in the data/ directory")
        return False

def validate_data_file():
    """Basic validation of the CSV data file format."""
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        return False
    
    # Use the first CSV file found
    data_file = csv_files[0]
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for participant data
        participant_count = len([line for line in content.split('\n') 
                               if line.strip() and line.strip()[0].isdigit()])
        
        if participant_count >= 19:  # Should have ~20 participants
            print(f"✅ Data validation passed: Found {participant_count} participant records in {data_file.name}")
            return True
        else:
            print(f"⚠️  Data validation warning: Only found {participant_count} participant records in {data_file.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading data file {data_file.name}: {e}")
        return False

def main():
    """Main setup function."""
    print("LLM Study Analysis - Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install requirements
    print("\nInstalling requirements...")
    if not install_requirements():
        print("\n⚠️  Please install requirements manually before running the analysis.")
    
    # Check for data file
    print("\nChecking for data file...")
    data_exists = check_data_file()
    
    if data_exists:
        print("\nValidating data file...")
        validate_data_file()
    
    # Summary
    print("\n" + "=" * 40)
    print("Setup Summary:")
    print(f"✅ Python version compatible")
    print(f"✅ Directories created") 
    print(f"{'✅' if data_exists else '❌'} Data file {'found' if data_exists else 'missing'}")
    
    if data_exists:
        print("\n🎉 Setup complete! You can now run the analysis:")
        print("   python analyze_study_data.py")
    else:
        print("\n📝 Next steps:")
        print("   1. Place your CSV file in the data/ directory (any name is fine)")
        print("   2. Run: python analyze_study_data.py")
    
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main()