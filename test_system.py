"""
Test script to verify the braindecode training system works correctly.
"""

import os
import sys

def test_imports():
    """Test if all required imports work correctly."""
    print("Testing imports...")
    
    try:
        import braindecode
        print(f"✓ braindecode version: {braindecode.__version__}")
    except ImportError as e:
        print(f"✗ braindecode import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        import mne
        print(f"✓ mne version: {mne.__version__}")
    except ImportError as e:
        print(f"✗ mne import failed: {e}")
        return False
    try:
        from PyQt5 import Qt
        print(f"✓ PyQt5 version: {Qt.PYQT_VERSION_STR}")
    except ImportError as e:
        print(f"✗ PyQt5 import failed: {e}")
        return False
    
    try:
        from braindecode.models import EEGNetv4, ShallowFBCSPNet
        print("✓ braindecode models imported successfully")
    except ImportError as e:
        print(f"✗ braindecode models import failed: {e}")
        return False
    
    try:
        import matplotlib
        import seaborn
        print("✓ visualization libraries imported successfully")
    except ImportError as e:
        print(f"✗ visualization libraries import failed: {e}")
        return False
    
    return True

def test_data_path():
    """Test if the data path exists."""
    data_path = r"C:\Users\Chari\OneDrive\Documentos\GitHub\projetoBCI\eeg_data"
    print(f"\nTesting data path: {data_path}")
    
    if os.path.exists(data_path):
        print("✓ Data path exists")
        
        # Check for subject directories
        eeg_path = os.path.join(data_path, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0')
        if os.path.exists(eeg_path):
            print("✓ EEG data structure found")
            
            # Count available subjects
            subjects = [d for d in os.listdir(eeg_path) if d.startswith('S') and os.path.isdir(os.path.join(eeg_path, d))]
            print(f"✓ Found {len(subjects)} subject directories")
            
            if len(subjects) > 0:
                # Check first subject for CSV files
                first_subject = os.path.join(eeg_path, subjects[0])
                csv_files = [f for f in os.listdir(first_subject) if f.endswith('.csv')]
                print(f"✓ Found {len(csv_files)} CSV files in {subjects[0]}")
                return True
            else:
                print("✗ No subject directories found")
                return False
        else:
            print("✗ EEG data structure not found")
            return False
    else:
        print("✗ Data path does not exist")
        return False

def test_basic_functionality():
    """Test basic functionality of the system components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test EEGDataManager
        from braindecode_training_system import EEGDataManager
        data_path = r"C:\Users\Chari\OneDrive\Documentos\GitHub\projetoBCI\eeg_data"
        
        if os.path.exists(data_path):
            data_manager = EEGDataManager(data_path)
            print("✓ EEGDataManager created successfully")
            
            # Try to load one subject
            try:
                subject_data = data_manager.load_subject_data(1)
                if subject_data is not None:
                    print(f"✓ Loaded subject 1 data: {subject_data['n_trials']} trials")
                else:
                    print("✗ Failed to load subject 1 data")
                    return False
            except Exception as e:
                print(f"✗ Error loading subject data: {e}")
                return False
        else:
            print("⚠ Skipping data loading test (no data path)")
        
        # Test BrainDecodeTrainer
        from braindecode_training_system import BrainDecodeTrainer
        trainer = BrainDecodeTrainer()
        print("✓ BrainDecodeTrainer created successfully")
        
        # Test model creation
        import numpy as np
        model = trainer.create_model('EEGNetv4', n_channels=16, n_samples=388, n_classes=2)
        print("✓ EEGNetv4 model created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Braindecode Training System Test ===\n")
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test data path
    success &= test_data_path()
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    print("\n" + "="*50)
    if success:
        print("✓ All tests passed! The system is ready to use.")
        print("\nTo run the GUI application, execute:")
        print("python braindecode_training_system.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
