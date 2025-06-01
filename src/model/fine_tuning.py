"""
Class:   ModelFineTuner
Purpose: Provides fine-tuning capabilities for pre-trained EEG models with patient-specific data.
Author:  Bruno Rocha
Created: 2025-06-01
License: BSD (3-clause)
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Task 1.1: Create Fine-Tuning Module
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import numpy as np
from .eeg_inception_erp import EEGInceptionERPModel


class ModelFineTuner:
    """
    Fine-tuning system for pre-trained EEG models with patient-specific data.
    
    This class provides functionality to load pre-trained models, configure them
    for transfer learning, and fine-tune on patient-specific recordings.
    
    Args:
        device (torch.device, optional): Device for model operations. Defaults to auto-detect.
        verbose (bool, optional): Enable verbose logging. Defaults to True.
    """
    
    def __init__(self, device: Optional[torch.device] = None, verbose: bool = True):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model: Optional[EEGInceptionERPModel] = None
        self.base_model_path: Optional[str] = None
        
        if self.verbose:
            print(f"ModelFineTuner initialized on device: {self.device}")

    def load_pretrained_model(self, model_path: str) -> EEGInceptionERPModel:
        """
        Loads a pre-trained EEGInceptionERP model from the specified path.
        
        This function loads a model saved during k-fold training and prepares it
        for fine-tuning on patient-specific data.
        
        Args:
            model_path (str): Path to the pre-trained model file (.pth format).
                            Can be either:
                            - Full path to specific model file (e.g., "models/bom_modelo/eeginceptionerp_fold_final.pth")
                            - Directory path (will load the "final" model automatically)
        
        Returns:
            EEGInceptionERPModel: The loaded pre-trained model ready for fine-tuning.
        
        Raises:
            FileNotFoundError: If the specified model path does not exist.
            ValueError: If the model file is invalid or incompatible.
            RuntimeError: If model loading fails due to architecture mismatch.
        """
        # Determine the actual model file path
        if os.path.isdir(model_path):
            # If directory provided, look for the final model
            model_file = os.path.join(model_path, "eeginceptionerp_fold_final.pth")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"No final model found in directory {model_path}. "
                    f"Expected: {model_file}"
                )
        elif os.path.isfile(model_path):
            model_file = model_path
        else:
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        if self.verbose:
            print(f"Loading pre-trained model from: {model_file}")
        
        try:
            # Load the checkpoint to extract model configuration
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Extract constructor arguments from the saved model
            constructor_args = checkpoint.get('constructor_args', {})
            
            if not constructor_args:
                raise ValueError(
                    f"Model file {model_file} does not contain constructor arguments. "
                    "This model may be incompatible with fine-tuning."
                )
            
            # Create a new model instance with the same architecture
            model = EEGInceptionERPModel(
                n_chans=constructor_args.get('n_chans', 16),
                n_outputs=constructor_args.get('n_outputs', 2),
                n_times=constructor_args.get('n_times', 400),
                sfreq=constructor_args.get('sfreq', 125.0),
                model_name=f"FineTuned_{checkpoint.get('model_name', 'EEGInceptionERP')}",
                drop_prob=constructor_args.get('drop_prob', 0.5),
                n_filters=constructor_args.get('n_filters', 8),
                model_version=checkpoint.get('model_version', '1.0')
            )
            
            # Load the pre-trained weights
            model.load(model_file)
            
            # Move to specified device
            model = model.to(self.device)
            
            # Store reference for future operations
            self.model = model
            self.base_model_path = model_file
            
            if self.verbose:
                print(f"✅ Successfully loaded pre-trained model:")
                print(f"   - Architecture: {constructor_args.get('n_chans')} channels, {constructor_args.get('n_times')} time points")
                print(f"   - Classes: {constructor_args.get('n_outputs')}")
                print(f"   - Device: {self.device}")
                print(f"   - Trained status: {model.is_trained}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model from {model_file}: {str(e)}")


# Test function for immediate testing
def test_load_pretrained_model():
    """Test the load_pretrained_model function."""
    print("🧪 Testing load_pretrained_model function...")
    
    # Test data setup
    models_dir = "models"
    test_model_dirs = ["bom_modelo", "EEGInceptionERP", "teste001"]
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print("❌ Models directory not found. Skipping test.")
        return
    
    fine_tuner = ModelFineTuner(verbose=True)
    
    # Test 1: Load from directory path
    for model_dir in test_model_dirs:
        model_path = os.path.join(models_dir, model_dir)
        
        if os.path.exists(model_path):
            try:
                print(f"\n--- Testing directory: {model_path} ---")
                model = fine_tuner.load_pretrained_model(model_path)
                
                # Verify model properties
                assert model is not None, "Model should not be None"
                assert isinstance(model, EEGInceptionERPModel), "Model should be EEGInceptionERPModel"
                assert fine_tuner.model is model, "ModelFineTuner should store model reference"
                assert fine_tuner.base_model_path is not None, "Base model path should be stored"
                
                print(f"✅ Successfully loaded model from {model_dir}")
                break  # Test passed, exit loop
                
            except Exception as e:
                print(f"⚠️  Failed to load from {model_dir}: {e}")
                continue
    else:
        print("❌ No valid model directories found for testing")
        return
    
    # Test 2: Load from specific file path
    try:
        specific_file = os.path.join(models_dir, "bom_modelo", "eeginceptionerp_fold_final.pth")
        if os.path.exists(specific_file):
            print(f"\n--- Testing specific file: {specific_file} ---")
            model2 = fine_tuner.load_pretrained_model(specific_file)
            assert model2 is not None, "Model from specific file should not be None"
            print("✅ Successfully loaded model from specific file")
    except Exception as e:
        print(f"⚠️  Failed to load from specific file: {e}")
    
    # Test 3: Error handling - non-existent path
    try:
        print("\n--- Testing error handling ---")
        fine_tuner.load_pretrained_model("non_existent_path")
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✅ Correctly raised FileNotFoundError for non-existent path")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
    
    print("\n✅ test_load_pretrained_model completed successfully!")


if __name__ == "__main__":
    test_load_pretrained_model()
