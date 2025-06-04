"""
Class:   Model (Abstract Base Class)
Purpose: Defines the interface for all machine learning models in the BCI project.
Author:  Bruno Rocha
Created: 2025-05-28
License: BSD (3-clause) # Verify actual license
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         This ABC ensures that all specific model implementations adhere to a
         common structure, facilitating modularity and interchangeability.
"""
from abc import ABC, abstractmethod
import torch

class Model(ABC, torch.nn.Module): # Inherit from torch.nn.Module if models are PyTorch based
    """
    Abstract base class for all machine learning models in the BCI project.

    This class defines the common interface that all specific model implementations
    (e.g., EEGInceptionERPModel) must adhere to. It ensures
    modularity and interchangeability of models within the system.

    If models are PyTorch-based, they should also inherit from `torch.nn.Module`.

    Args:
        model_name (str): The specific name of the model instance.
        model_version (str, optional): Version of the model. Defaults to "1.0".
    """

    def __init__(self, model_name: str, model_version: str = "1.0"):
        super().__init__() # Call torch.nn.Module's init
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")
        if not model_version or not isinstance(model_version, str):
            raise ValueError("model_version must be a non-empty string.")

        self._model_name = model_name
        self._model_version = model_version
        self._is_trained = False # Internal flag

    @property
    def name(self) -> str:
        """
        Returns the name of the model.
        """
        return self._model_name

    @property
    def version(self) -> str:
        """
        Returns the version of the model.
        """
        return self._model_version

    @property
    def is_trained(self) -> bool:
        """
        Indicates if the model has been trained.
        """
        return self._is_trained

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        This method must be implemented by all concrete model subclasses.
        For non-PyTorch models, this might be adapted or named differently (e.g., `predict_proba`).

        Args:
            x (torch.Tensor): Input tensor. Shape depends on the specific model.

        Returns:
            torch.Tensor: Output tensor from the model. Shape depends on the specific model.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Saves the model's state to a file.

        Args:
            file_path (str): The path (including filename) where the model should be saved.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            IOError: If there's an issue writing the file.
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        """
        Loads the model's state from a file.

        Args:
            file_path (str): The path (including filename) from which to load the model.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            FileNotFoundError: If the specified file_path does not exist.
            IOError: If there's an issue reading the file.
        """
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Gets the torch.device where the model's parameters are allocated.
        For non-PyTorch models, this might return None or raise an error.

        Returns:
            torch.device: The device (e.g., 'cpu' or 'cuda:0') of the model.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
            RuntimeError: If the model has no parameters or device cannot be determined.
        """
        pass

    def summary(self) -> str:
        """
        Provides a brief summary of the model.
        Can be overridden by subclasses for more specific information.

        Returns:
            str: A string summary of the model.
        """
        return f"Model: {self.name} (Version: {self.version}, Trained: {self.is_trained})"

# Example of how a non-PyTorch model might look (conceptual)
# class NonTorchModel(Model):
#     def __init__(self, model_name: str, model_version: str = "1.0"):
#         # Need to handle super() differently if not using torch.nn.Module
#         # For now, this example assumes we find a way to bypass torch.nn.Module's __init__
#         # or Model's __init__ is adjusted.
#         # super(Model, self).__init__() # This would skip torch.nn.Module's init
#         self._model_name = model_name
#         self._model_version = model_version
#         self._is_trained = False
#         # ... other init for non-torch model

#     def forward(self, x): # Signature might differ
#         raise NotImplementedError("NonTorchModel does not use PyTorch forward.")

#     def predict(self, data):
#         # Actual prediction logic
#         pass
    
#     def save(self, file_path: str):
#         # Save logic for sklearn, etc.
#         pass

#     def load(self, file_path: str):
#         # Load logic
#         pass

#     @property
#     def device(self):
#         return None # Or raise an error

# __all__ = ['Model']
