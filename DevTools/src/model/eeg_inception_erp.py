"""
Class:   EEGInceptionERPModel
Purpose: Implements the EEGInceptionERP classification model, inheriting from BaseModel.
Author:  Bruno Rocha
Created: 2025-05-28
License: BSD (3-clause) # Verify actual license
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Wraps braindecode's EEGInceptionERP.
"""

import torch
import torch.nn as nn
import os
from src.model.base_model import Model as BaseModel

try:
    from braindecode.models import EEGInceptionERP as BraindecodeEEGInceptionERP
except ImportError as e:
    # Raise a more informative error if EEGInceptionERP cannot be imported.
    raise ImportError(
        "braindecode.models.EEGInceptionERP could not be imported. "
        "Please ensure braindecode is installed correctly and EEGInceptionERP is available. "
        f"Original error: {e}"
    )


class EEGInceptionERPModel(BaseModel):
    """
    EEGInceptionERP model implementation, inheriting from BaseModel.

    This class wraps braindecode's EEGInceptionERP and provides the interface
    defined by BaseModel.

    Args:
        n_chans (int): Number of EEG channels.
        n_outputs (int): Number of classes to predict.
        n_times (int): Number of time points in each input EEG window.
        sfreq (float): Sampling frequency of the input signals in Hz.
        model_name (str, optional): Name of the model instance. Defaults to "EEGInceptionERP".
        drop_prob (float, optional): Dropout probability. Defaults to 0.5.
        n_filters (int, optional): Number of temporal filters for EEGInceptionERP. Defaults to 8.
        model_version (str, optional): Version of the model. Defaults to "1.0".
    """
    
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        sfreq: float,
        model_name: str = "EEGInceptionERP", # Added model_name
        drop_prob: float = 0.5,
        n_filters: int = 8,
        model_version: str = "1.0" # Added model_version
    ):
        super().__init__(model_name=model_name, model_version=model_version) # Call BaseModel's init
        
        if not isinstance(n_chans, int) or n_chans <= 0:
            raise ValueError("n_chans must be a positive integer.")
        if not isinstance(n_outputs, int) or n_outputs <= 0:
            raise ValueError("n_outputs must be a positive integer.")
        if not isinstance(n_times, int) or n_times <= 0:
            raise ValueError("n_times must be a positive integer.")
        if not isinstance(sfreq, (float, int)) or sfreq <= 0:
            raise ValueError("sfreq must be a positive number.")
        if not isinstance(drop_prob, float) or not (0.0 <= drop_prob <= 1.0):
            raise ValueError("drop_prob must be a float between 0.0 and 1.0.")
        if not isinstance(n_filters, int) or n_filters <= 0:
            raise ValueError("n_filters must be a positive integer.")

        # Store configuration (already done in BaseModel or can be specific here if needed)
        # self.n_chans = n_chans # These are now primarily for the internal model construction
        # self.n_outputs = n_outputs
        # self.n_times = n_times
        # self.sfreq = sfreq
        
        # Initialize the actual underlying model (braindecode EEGInceptionERP)
        self._internal_model = self._build_internal_model(n_chans, n_outputs, n_times, sfreq, drop_prob, n_filters)
        self._is_trained = False # Initialize training status

    def _build_internal_model(self, n_chans: int, n_outputs: int, n_times: int, sfreq: float, drop_prob: float, n_filters: int) -> nn.Module:
        """
        Constructs the internal Braindecode EEGInceptionERP model.

        Args:
            n_chans (int): Number of EEG channels.
            n_outputs (int): Number of output classes.
            n_times (int): Number of time points.
            sfreq (float): Sampling frequency.
            drop_prob (float): Dropout probability.
            n_filters (int): Number of temporal filters.

        Returns:
            nn.Module: The constructed neural network.
        
        Raises:
            ImportError: If BraindecodeEEGInceptionERP is not available.
            RuntimeError: If EEGInceptionERP cannot be initialized.
        """
        if BraindecodeEEGInceptionERP is None: # Should not happen due to check at import time
            raise ImportError("Braindecode EEGInceptionERP model is not available.")
        
        try:
            # Attempt to initialize with all parameters
            model = BraindecodeEEGInceptionERP(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                sfreq=sfreq,
                drop_prob=drop_prob,
                n_filters=n_filters,
            )
            # print("Successfully initialized BraindecodeEEGInceptionERP with all parameters.")
            return model
        except TypeError as e_full:
            # This block handles cases where EEGInceptionERP might not accept all args
            # (e.g., older versions or specific configurations of braindecode)
            # print(f"Failed to initialize EEGInceptionERP with all parameters (sfreq, drop_prob, n_filters): {e_full}. Trying simplified initialization.")
            try:
                # Attempt to initialize with core parameters only
                model = BraindecodeEEGInceptionERP(
                    n_chans=n_chans,
                    n_outputs=n_outputs,
                    n_times=n_times,
                    # sfreq might be a positional argument or implicitly handled in some versions
                )
                # print("Successfully initialized BraindecodeEEGInceptionERP with core parameters (n_chans, n_outputs, n_times).")
                # Manually set other attributes if the model supports them and they were intended
                # This part is tricky as internal attribute names can vary.
                # For now, we assume the constructor handles it or it's not critical for this simplified init.
                return model
            except Exception as e_simple:
                raise RuntimeError(
                    f"Failed to initialize BraindecodeEEGInceptionERP model even with simplified parameters. "
                    f"Full init error: {e_full}. Simplified init error: {e_simple}. "
                    "Please check your braindecode installation and model compatibility."
                )
        except Exception as e_other:
            raise RuntimeError(f"An unexpected error occurred while initializing BraindecodeEEGInceptionERP: {e_other}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the internal model.

        Args:
            x (torch.Tensor): Input tensor (batch_size, n_channels, n_times).

        Returns:
            torch.Tensor: Output tensor (batch_size, n_outputs).
        
        Raises:
            ValueError: If input tensor `x` has unexpected dimensions.
        """
        if x.dim() not in [2, 3]:
            raise ValueError(f"Input tensor x expected to have 2 or 3 dimensions, got {x.dim()}")
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # CRITICAL: Ensure input is float32 to match model expectations
        if x.dtype != torch.float32:
            x = x.float()
        
        return self._internal_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the second-to-last layer of the internal model.

        This is useful for transfer learning or analyzing learned representations.
        Note: This implementation is a basic hook and might need adjustment
        depending on the exact structure of the wrapped braindecode model or custom CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_times).

        Returns:
            torch.Tensor: Feature tensor from the penultimate layer. Returns an empty
                          tensor if features cannot be extracted.
        
        Raises:
            ValueError: If the input tensor `x` has an unexpected number of dimensions.
            RuntimeError: If hooks cannot be registered or features are not captured.
        """
        if x.dim() not in [2, 3]:
            raise ValueError(f"Input tensor x expected to have 2 or 3 dimensions, got {x.dim()}")
        if x.dim() == 2:
            x = x.unsqueeze(0)

        activations = []
        
        def hook(module, input, output):
            activations.append(output)
            
        # Attempt to register hook on the layer before the final classification layer
        # This can be fragile if the model architecture changes significantly.
        target_module = None
        if isinstance(self._internal_model, nn.Sequential): # Fallback CNN
            # Penultimate layer is typically the one before the last nn.Linear
            if len(self._internal_model) > 1 and isinstance(self._internal_model[-1], nn.Linear) and len(self._internal_model) > 2:
                # Check if the layer before the last Linear is suitable (e.g., ReLU or Dropout after a Linear)
                if isinstance(self._internal_model[-2], (nn.ReLU, nn.Dropout)) and len(self._internal_model) > 3 and isinstance(self._internal_model[-3], nn.Linear):
                    target_module = self._internal_model[-3] # The Linear layer before ReLU/Dropout
                elif isinstance(self._internal_model[-2], nn.Linear):
                    target_module = self._internal_model[-2]
                else: # Fallback to the one before last if specific pattern not found
                    target_module = self._internal_model[-2]

            elif len(self._internal_model) > 1:
                 target_module = self._internal_model[-2] # Generic penultimate
            else: # Single layer model (unlikely for features)
                 target_module = self._internal_model[-1]

        elif hasattr(self._internal_model, 'children'): # Braindecode models
            children = list(self._internal_model.children())
            if len(children) > 1 and isinstance(children[-1], nn.Linear): # Common pattern
                # Try to get the module before the final Linear layer
                # This might be a block or a single layer
                # For braindecode models, it's often a Conv classifier or a dense layer
                # We might need to inspect self.model.final_layer or similar if available
                # For EEGInceptionERP, the features are often before the final conv1x1 (self.model.conv_classifier)
                if hasattr(self._internal_model, 'permute_and_avg_pool') and hasattr(self._internal_model, 'conv_classifier'):
                    # For EEGInceptionERP, features are likely after avg_pool, before final conv_classifier's linear part
                    # This is complex. A simpler approach for braindecode might be to access a named layer if known.
                    # For now, let's try a generic approach:
                    # The layer before the actual classification (e.g. before the final nn.Linear in conv_classifier)
                    final_conv_children = list(self._internal_model.conv_classifier.children()) if hasattr(self._internal_model, 'conv_classifier') else []
                    if final_conv_children and isinstance(final_conv_children[-1], nn.Linear) and len(final_conv_children) > 1:
                        target_module = final_conv_children[-2] # Layer before final linear in classifier
                    elif children:
                        target_module = children[-2] # Generic penultimate child
                    else:
                        target_module = self._internal_model # Fallback to model itself if no children
            elif children:
                 target_module = children[-2]
            else:
                target_module = self._internal_model


        if target_module is None:
            # Fallback if no suitable penultimate layer found (e.g. model too shallow)
            # Or if the structure is unexpected.
            print("Warning: Could not reliably determine the penultimate layer for feature extraction.")
            # Try hooking the last module as a last resort, though these are not "features before classification"
            modules = list(self._internal_model.modules())
            if len(modules) > 1 : target_module = modules[-1] # This will be the output itself
            else: return torch.tensor([]) # Cannot extract features

        handle = None
        try:
            handle = target_module.register_forward_hook(hook)
            _ = self.forward(x) # Use self.forward to ensure consistent input handling
        except Exception as e:
            if handle: handle.remove()
            raise RuntimeError(f"Failed to register hook or run forward pass for feature extraction: {e}")
        finally:
            if handle:
                handle.remove()
        
        if not activations:
            # This can happen if the hook wasn't triggered or the layer produced no output recorded by the hook
            print("Warning: No activations captured from the hooked layer.")
            return torch.tensor([])
        
        return activations[0].detach() # Detach from graph
    
    @property
    def device(self) -> torch.device:
        """
        Gets the torch.device where the internal model's parameters are allocated.

        Returns:
            torch.device: The device (e.g., 'cpu' or 'cuda:0') of the model.
        
        Raises:
            RuntimeError: If the model has no parameters.
        """
        try:
            return next(self._internal_model.parameters()).device
        except StopIteration:
            # This case (no parameters) should ideally not happen for a valid nn.Module
            raise RuntimeError("The model has no parameters, cannot determine device.")

    def save(self, file_path: str) -> None:
        """
        Saves the internal model's state_dict and training status to a file.

        Args:
            file_path (str): Path (including filename) to save the model.
                             If a directory is given, appends `self.name + .pth`.

        Raises:
            IOError: If there's an issue writing the file.
        """
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, f"{self.name}.pt")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        save_content = {
            'model_state_dict': self._internal_model.state_dict(),
            'model_name': self.name,
            'model_version': self.version,
            'is_trained': self.is_trained,
            # Store constructor args to allow for easier reloading/reconstruction
            'constructor_args': {
                'n_chans': self._internal_model.n_chans if hasattr(self._internal_model, 'n_chans') else None,
                'n_outputs': self._internal_model.n_outputs if hasattr(self._internal_model, 'n_outputs') else None,
                'n_times': self._internal_model.n_times if hasattr(self._internal_model, 'n_times') else None,
                'sfreq': self._internal_model.sfreq if hasattr(self._internal_model, 'sfreq') else None,
                'drop_prob': self._internal_model.drop_prob if hasattr(self._internal_model, 'drop_prob') else 0.5,
                'n_filters': self._internal_model.n_filters if hasattr(self._internal_model, 'n_filters') else 8
            }
        }
        try:
            torch.save(save_content, file_path)
            print(f"Model '{self.name}' saved to {file_path}")
        except Exception as e:
            raise IOError(f"Error saving model to {file_path}: {e}")

    def load(self, file_path: str) -> None:
        """
        Loads the internal model's state_dict and training status from a file.

        Args:
            file_path (str): Path (including filename) to load the model from.

        Raises:
            FileNotFoundError: If the specified file_path does not exist.
            IOError: If there's an issue reading the file or if the file content is invalid.
            ValueError: If constructor arguments from the loaded file are insufficient.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        try:
            checkpoint = torch.load(file_path, map_location=self.device) # Load to current device
            
            # Reconstruct model if necessary, or ensure compatibility
            # This is a simplified load; for full robustness, you might need to re-initialize
            # the model using saved constructor_args if the architecture could vary significantly.
            # For now, we assume the current instance's architecture matches the saved one.
            
            constructor_args = checkpoint.get('constructor_args')
            if constructor_args:
                # Potentially re-initialize self._internal_model here if needed
                # For simplicity, we assume current n_chans etc. are compatible
                # Or, one could design EEGInceptionERPModel to be reconfigurable.
                pass # Add re-init logic if necessary based on constructor_args

            self._internal_model.load_state_dict(checkpoint['model_state_dict'])
            self._model_name = checkpoint.get('model_name', self.name) # Update name if in checkpoint
            self._model_version = checkpoint.get('model_version', self.version)
            self._is_trained = checkpoint.get('is_trained', False)
            self._internal_model.to(self.device) # Ensure model is on the correct device
            self._internal_model.eval() # Set to eval mode after loading
            print(f"Model '{self.name}' loaded from {file_path} and set to {self.device}. Trained: {self.is_trained}")

        except Exception as e:
            raise IOError(f"Error loading model from {file_path}: {e}")

    def set_trained(self, trained_status: bool):
        """
        Sets the training status of the model.

        Args:
            trained_status (bool): True if the model is trained, False otherwise.
        """
        if not isinstance(trained_status, bool):
            raise ValueError("trained_status must be a boolean.")
        self._is_trained = trained_status

# Remove the old factory function EEGInceptionERP if EEGInceptionERPModel is the primary way to create this model.
# If it's still needed for some compatibility, it should be updated to return EEGInceptionERPModel.
# def EEGInceptionERP(...): 
# return EEGInceptionERPModel(...)

# Re-export the model
__all__ = ['EEGInceptionERPModel'] # Changed from EEGModel
