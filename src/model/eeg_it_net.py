"""
Class:   EEGITNetModel
Purpose: Wraps the EEGITNet model from the braindecode library, providing a
         standardized interface for training, evaluation, and feature extraction
         within the BCI project.
Author:  Bruno Rocha
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Implements the BaseModel interface.
"""
import os
import torch
import torch.nn as nn # Ensure nn is imported if not already for new params
from typing import Optional, Dict, Any, Tuple

try:
    from braindecode.models import EEGITNet as BraindecodeEEGITNet
except ImportError:
    BraindecodeEEGITNet = None # Allow for environments where braindecode might not be installed

from .base_model import Model as BaseModel # Corrected import

class EEGITNetModel(BaseModel):
    """
    Wrapper for the braindecode.models.EEGITNet model.

    This class provides a consistent API for using EEGITNet within the BCI
    application, conforming to the BaseModel interface. It handles model
    initialization, forward pass, feature extraction, saving, and loading.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: Optional[int] = None,
        sfreq: Optional[float] = None,
        drop_prob: float = 0.5, # Default based on common Braindecode usage
        add_log_softmax: bool = True, # Common default in Braindecode
        model_name: str = "EEGITNet",
        model_version: str = "1.0"
    ):
        """
        Initializes the EEGITNetModel.

        Args:
            n_chans (int): Number of EEG channels. Must be positive.
            n_outputs (int): Number of output classes. Must be positive.
            n_times (Optional[int]): Number of time samples in the input.
                Must be positive if provided.
            sfreq (Optional[float]): Sampling frequency of the EEG data [Hz].
                Must be positive if provided.
            drop_prob (float): Dropout probability. Must be between 0.0 and 1.0.
            add_log_softmax (bool): Whether to use log-softmax non-linearity as the output.
            model_name (str): Name of the model instance.
            model_version (str): Version of the model instance.

        Raises:
            ValueError: If any input parameters are invalid.
            ImportError: If braindecode is not installed.
            RuntimeError: If EEGITNet model initialization fails.
        """
        super().__init__(model_name, model_version)

        if BraindecodeEEGITNet is None:
            raise ImportError(
                "Braindecode library is not installed. "
                "Please install it to use EEGITNetModel (e.g., `pip install braindecode`)."
            )

        # Input validation
        if not isinstance(n_chans, int) or n_chans <= 0:
            raise ValueError("n_chans must be a positive integer.")
        if not isinstance(n_outputs, int) or n_outputs <= 0:
            raise ValueError("n_outputs must be a positive integer.")
        if n_times is not None and (not isinstance(n_times, int) or n_times <= 0):
            raise ValueError("If provided, n_times must be a positive integer.")
        if sfreq is not None and (not isinstance(sfreq, (int, float)) or sfreq <= 0):
            raise ValueError("If provided, sfreq must be a positive number.")
        if not (isinstance(drop_prob, float) and 0.0 <= drop_prob <= 1.0):
            raise ValueError("drop_prob must be a float between 0.0 and 1.0.")
        if not isinstance(add_log_softmax, bool):
            raise ValueError("add_log_softmax must be a boolean.")

        # Removed activation_map and _activation_module as 'activation' param is removed

        self._n_chans = n_chans
        self._n_outputs = n_outputs
        self._n_times = n_times
        self._sfreq = sfreq
        
        # Store all original parameters for saving/loading and reconstruction
        self._constructor_args = {
            "n_chans": n_chans,
            "n_outputs": n_outputs,
            "n_times": n_times,
            "sfreq": sfreq,
            "drop_prob": drop_prob,
            "add_log_softmax": add_log_softmax,
            "model_name": model_name,
            "model_version": model_version,
        }

        # Arguments for BraindecodeEEGITNet constructor
        # Pass only parameters expected by the simplified/actual EEGITNet signature
        self._eegitnet_kwargs = {
            "drop_prob": drop_prob,
            "add_log_softmax": add_log_softmax,
            # Other specific EEGITNet hyperparams (kernel_length, etc.) are removed
        }
        
        # n_chans, n_outputs, n_times, sfreq are passed directly to _build_internal_model
        # and then to BraindecodeEEGITNet

        self._internal_model = self._build_internal_model(
            n_chans=self._n_chans,
            n_outputs=self._n_outputs,
            n_times=self._n_times,
            sfreq=self._sfreq,
            **self._eegitnet_kwargs
        )
        self._is_trained = False

    def _build_internal_model(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: Optional[int], # Added n_times
        sfreq: Optional[float], # Added sfreq
        **kwargs: Any
    ) -> nn.Module:
        """
        Constructs the internal EEGITNet model.

        Args:
            n_chans (int): Number of EEG channels.
            n_outputs (int): Number of output classes.
            n_times (Optional[int]): Number of time samples.
            sfreq (Optional[float]): Sampling frequency.
            **kwargs: Additional keyword arguments for BraindecodeEEGITNet.

        Returns:
            nn.Module: The initialized EEGITNet model.

        Raises:
            RuntimeError: If EEGITNet model initialization fails.
        """
        try:
            # Braindecode's EEGITNet (and EEGModuleMixin) expects n_chans, n_outputs,
            # n_times, sfreq directly in its __init__ or through super()
            model = BraindecodeEEGITNet(
                n_chans=n_chans,
                n_outputs=n_outputs, # EEGITNet uses n_outputs
                n_times=n_times,
                sfreq=sfreq,
                **kwargs # Pass the specific EEGITNet params
            )
            return model
        except Exception as e:
            # Add more context to the error message
            import traceback
            tb_str = traceback.format_exc()
            raise RuntimeError(f"Failed to initialize BraindecodeEEGITNet: {e}\\nBraindecode Traceback:\\n{tb_str}\\nPassed kwargs: {kwargs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the internal EEGITNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_times)
                              or (n_channels, n_times).

        Returns:
            torch.Tensor: Output tensor (batch_size, n_outputs).
        
        Raises:
            ValueError: If input tensor `x` has unexpected dimensions or type.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor, got {type(x)}")
        if x.dim() not in [2, 3]:
            raise ValueError(
                f"Input tensor x expected to have 2 or 3 dimensions, got {x.dim()}"
            )
        if x.dim() == 2:
            x = x.unsqueeze(0) # Add batch dimension if missing
        
        x = x.to(self.device)
        return self._internal_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the EEGITNet model, typically before the final classifier.
        For EEGITNet, this might involve accessing a specific layer's output.
        This implementation attempts to get features from a layer named 'feature_extractor'
        or a common penultimate layer if 'feature_extractor' is not standard.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_times)
                              or (n_channels, n_times).

        Returns:
            torch.Tensor: Feature tensor. Returns an empty tensor if features cannot be extracted.
        
        Raises:
            ValueError: If input tensor `x` has unexpected dimensions or type.
            RuntimeError: If hooks cannot be registered or features are not captured.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor, got {type(x)}")
        if x.dim() not in [2, 3]:
            raise ValueError(
                f"Input tensor x expected to have 2 or 3 dimensions, got {x.dim()}"
            )
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        activations = []
        hook_handle = None

        # Braindecode's EEGITNet has a .feature_extractor attribute which is an nn.Sequential
        # The output of this is typically what we want.
        if hasattr(self._internal_model, 'feature_extractor') and \
           isinstance(self._internal_model.feature_extractor, nn.Module):
            target_module = self._internal_model.feature_extractor
        else:
            # Fallback: try to find a common penultimate layer (e.g., before a final Linear layer)
            # This is more fragile and model-specific.
            # For EEGITNet, the feature_extractor is the standard way.
            print("Warning: EEGITNetModel.get_features: `feature_extractor` module not found "
                  "or not an nn.Module. Cannot extract features reliably this way.")
            return torch.tensor([], device=self.device)

        def hook(module, input_val, output_val):
            activations.append(output_val)

        try:
            hook_handle = target_module.register_forward_hook(hook)
            with torch.no_grad():
                self._internal_model(x) # Run forward pass to trigger hook
        except Exception as e:
            if hook_handle: hook_handle.remove()
            raise RuntimeError(f"Failed to register hook or run forward pass for features: {e}")
        finally:
            if hook_handle:
                hook_handle.remove()
        
        if not activations:
            print("Warning: EEGITNetModel.get_features: No activations captured.")
            return torch.tensor([], device=self.device)
        
        # The output of feature_extractor might already be flattened or pooled.
        return activations[0].detach()

    @property
    def device(self) -> torch.device:
        """
        Gets the torch.device where the internal model's parameters are allocated.

        Returns:
            torch.device: The device (e.g., 'cpu' or 'cuda:0') of the model.
        
        Raises:
            RuntimeError: If the model has no parameters or is not initialized.
        """
        try:
            return next(self._internal_model.parameters()).device
        except StopIteration: # No parameters in the model
            # This can happen if the model is empty or not properly initialized
            # Fallback to CPU if no parameters, though an initialized model should have them.
            # print("Warning: Model has no parameters. Assuming CPU device.")
            # return torch.device('cpu')
            raise RuntimeError("The model has no parameters, cannot determine device.")
        except AttributeError:
             raise RuntimeError("Internal model not properly initialized or not an nn.Module.")


    def save(self, file_path: str) -> None:
        """
        Saves the model's state_dict, training status, and constructor arguments.

        Args:
            file_path (str): Path to save the model. Appends `self.name + .pth` if a directory.

        Raises:
            IOError: If there's an issue writing the file.
            ValueError: If file_path is invalid.
        """
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("file_path must be a non-empty string.")

        # If file_path is a directory, create a filename
        if os.path.isdir(file_path):
            # Ensure the directory exists
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, self.name + ".pth")
        else:
            # Ensure the parent directory exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir: # If parent_dir is empty, it's a relative path in current dir
                os.makedirs(parent_dir, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self._internal_model.state_dict(),
            'is_trained': self._is_trained,
            'constructor_args': self._constructor_args, # Save constructor args
            'model_name': self.name,
            'model_version': self.version,
            'braindecode_version': None # Placeholder for braindecode version if needed
        }
        try:
            import braindecode
            save_dict['braindecode_version'] = braindecode.__version__
        except ImportError:
            pass # braindecode not available, version will remain None

        try:
            torch.save(save_dict, file_path)
            print(f"Model '{self.name}' saved to {file_path}")
        except IOError as e:
            raise IOError(f"Could not save model to {file_path}: {e}")
        except Exception as e: # Catch other potential torch.save errors
            raise RuntimeError(f"An unexpected error occurred while saving the model: {e}")

    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'EEGITNetModel':
        """
        Loads a model from a file.

        Args:
            file_path (str): Path to the saved model file.
            device (Optional[torch.device]): The device to load the model onto. 
                                             If None, uses the device from saved state or default.

        Returns:
            EEGITNetModel: The loaded model instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
            IOError: If there's an issue reading the file.
            RuntimeError: If model instantiation or state loading fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        try:
            checkpoint = torch.load(file_path, map_location=device if device else 'cpu')
        except IOError as e:
            raise IOError(f"Could not read model file {file_path}: {e}")
        except Exception as e: # Catch other torch.load errors (e.g., pickle, zip)
            raise RuntimeError(f"An unexpected error occurred while loading the model checkpoint: {e}")

        constructor_args = checkpoint.get('constructor_args')
        if constructor_args is None:
            raise RuntimeError("Invalid model file: constructor_args not found in checkpoint.")

        # Ensure all necessary args for __init__ are present
        required_args = [
            "n_chans", "n_outputs", "n_times", "sfreq", "drop_prob", "add_log_softmax"
        ]
        missing_args = [arg for arg in required_args if arg not in constructor_args]
        if missing_args:
            raise RuntimeError(f"Invalid model file: Missing constructor arguments: {', '.join(missing_args)}")
        
        # Optional args for __init__
        constructor_args.setdefault("model_name", checkpoint.get("model_name", "EEGITNetLoaded"))
        constructor_args.setdefault("model_version", checkpoint.get("model_version", "1.0"))


        try:
            # Create a new model instance with saved constructor arguments
            loaded_model = cls(**constructor_args)
            
            # Load the state dict
            loaded_model._internal_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model._is_trained = checkpoint.get('is_trained', False)
            
            # If a specific device is requested, move the model to it
            if device:
                loaded_model._internal_model.to(device)
            
            print(f"Model '{loaded_model.name}' loaded from {file_path} (Braindecode version when saved: {checkpoint.get('braindecode_version', 'N/A')})")
            print(f"Model is now on device: {loaded_model.device}")
            return loaded_model
        except KeyError as e:
            raise RuntimeError(f"Invalid model file: Missing key {e} in checkpoint.")
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            raise RuntimeError(f"Failed to load model from checkpoint: {e}\\nTraceback:\\n{tb_str}")

    def set_trained(self, trained_status: bool) -> None:
        """
        Sets the training status of the model.

        Args:
            trained_status (bool): True if the model is trained, False otherwise.
        
        Raises:
            ValueError: If trained_status is not a boolean.
        """
        if not isinstance(trained_status, bool):
            raise ValueError("trained_status must be a boolean.")
        self._is_trained = trained_status
        print(f"Model '{self.name}' trained status set to: {self._is_trained}")
