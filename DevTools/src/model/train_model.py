"""
Class:   ModelTrainer (Conceptual - this file contains training orchestration)
Purpose: Trains EEG classification models using K-fold cross-validation.
Author:  Bruno Rocha
Created: 2025-05-28
License: BSD (3-clause) # Assuming BSD, verify actual license
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Orchestrates the training and validation of EEG models (e.g., EEGModel)
         including data loading, K-fold splitting, epoch training, validation,
         early stopping, model saving, and results visualization.
"""
import os
import torch
import numpy as np
import matplotlib # Import matplotlib like this
matplotlib.use('Agg') # Then set backend. Must be before importing pyplot.
import matplotlib.pyplot as plt # Then import pyplot
from torch.utils.data import DataLoader
from src.data.data_loader import BCIDataLoader
from src.model.eeg_inception_erp import EEGInceptionERPModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split, KFold
import argparse # Added argparse for CLI argument parsing

# Training parameters
BATCH_SIZE = 10
# NUM_EPOCHS = 50 # Will be passed as a parameter
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
# K_FOLDS = 5 # Will be passed as a parameter
TEST_SPLIT = 0.2  # Hold out test set before K-fold

def train_epoch(model: EEGInceptionERPModel, dataloader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """
    Performs a single training epoch for the given model.

    Args:
        model (EEGInceptionERPModel): The neural network model to train.
        dataloader (DataLoader): DataLoader providing training data batches.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        device (torch.device): The device (CPU or CUDA) to perform training on.

    Returns:
        tuple[float, float]: A tuple containing the average epoch loss and epoch accuracy.
    
    Raises:
        # This function primarily relies on PyTorch, exceptions would typically be
        # RuntimeError from PyTorch operations if inputs are malformed, device issues, etc.
        # Explicit custom exceptions are not raised here.
        pass
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        # CRITICAL: Ensure consistent dtypes before moving to device
        inputs = inputs.float()  # Ensure float32
        labels = labels.long()   # Ensure int64
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model: EEGInceptionERPModel, dataloader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> tuple[float, float]:
    """
    Validates the model on the given dataset.

    Args:
        model (EEGInceptionERPModel): The neural network model to validate.
        dataloader (DataLoader): DataLoader providing validation data batches.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device (CPU or CUDA) to perform validation on.

    Returns:
        tuple[float, float]: A tuple containing the average validation loss and validation accuracy.

    Raises:
        # Similar to train_epoch, relies on PyTorch for operational exceptions.
        pass
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # CRITICAL: Ensure consistent dtypes before moving to device
            inputs = inputs.float()  # Ensure float32
            labels = labels.long()   # Ensure int64
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc

def train_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_num: int | str, 
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    batch_size: int,
    model_base_path: str,
    model_name: str,
    model_params: dict | None = None
) -> tuple[float, dict]:
    """
    Trains and validates a model for a single fold of K-fold cross-validation or for final training.

    Args:
        X_train (np.ndarray): Training feature data (windows, channels, timepoints).
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation feature data.
        y_val (np.ndarray): Validation labels.
        fold_num (int | str): Identifier for the current fold (e.g., 1, 2, ..., or "final").
        device (torch.device): Device for training (e.g., torch.device("cuda")).
        num_epochs (int): Maximum number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
        batch_size (int): Number of samples per training batch.
        model_base_path (str): Base directory path where model files will be saved.
        model_name (str): Name of the model, used for creating a subdirectory within `model_base_path`.
        model_params (dict | None, optional): Additional keyword arguments for the EEGInceptionERPModel constructor. Defaults to None.

    Returns:
        tuple[float, dict]: A tuple containing:
            - best_val_acc (float): The best validation accuracy achieved during this fold/training.
            - training_history (dict): A dictionary containing lists of training losses,
              training accuracies, validation losses, and validation accuracies per epoch.
    
    Raises:
        ValueError: If `n_outputs` cannot be determined from `y_train` (e.g. empty `y_train`).
        FileNotFoundError: If `model_base_path` or `model_name` subdirectories cannot be created.
        # Other exceptions can be raised by PyTorch operations or EEGModel instantiation.
    """
    # CRITICAL: Ensure data types are consistent from the start
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    
    # Create dataloaders
    train_loader = DataLoader(list(zip(X_train, y_train)), 
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), 
                          batch_size=batch_size)
    
    # Create model
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    n_outputs = len(np.unique(y_train))

    if model_params is None:
        model_params = {}

    # Always use EEGInceptionERPModel
    model = EEGInceptionERPModel(
        n_chans=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=125, 
        model_name=f"{model_name}_fold_{fold_num}", # Model instance name can include fold
        **model_params
    ).to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Use learning_rate parameter
    
    # Initialize tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Ensure model_save_path exists (now model_base_path/model_name)
    # The specific fold model path will be constructed inside the loop
    # os.makedirs(model_save_path, exist_ok=True) # This was for the old structure

    print(f"\\nTraining Fold {fold_num}...")
    for epoch in range(num_epochs): # Use num_epochs parameter
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}") # Use num_epochs parameter
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model for this fold
            model.set_trained(True)
            fold_model_save_path = os.path.join(model_base_path, model_name, f'eeginceptionerp_fold_{fold_num}.pt')
            os.makedirs(os.path.dirname(fold_model_save_path), exist_ok=True)
            model.save(fold_model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience: # Use early_stopping_patience parameter
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break
    
    training_history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    print(f"  Fold {fold_num} completed - Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc, training_history

def main(
    subjects_to_use: list[int] | str | None = None,
    num_epochs_per_fold: int = 30,
    num_k_folds: int = 10,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 8,
    batch_size: int = 10,
    test_split_ratio: float = 0.2,
    data_base_path: str = "eeg_data",
    runs_to_include: list[int] | None = None,
    model_name: str = "unnamed_model",
    model_params_json: str | None = None
) -> dict:
    """
    Main function to orchestrate the EEG model training and evaluation pipeline.

    This function loads data, performs K-fold cross-validation, trains a final model
    (EEGInceptionERP), and generates plots summarizing the training results.

    Args:
        subjects_to_use (list[int] | str | None, optional): List of subject IDs (integers),
            the string "all", or None. If None, defaults to subjects 1-10. Defaults to None.
        num_epochs_per_fold (int, optional): Max epochs for each fold. Defaults to 50.
        num_k_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        learning_rate (float, optional): Optimizer learning rate. Defaults to 0.001.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 5.
        batch_size (int, optional): Training batch size. Defaults to 32.
        test_split_ratio (float, optional): Proportion of data for the hold-out test set (0.0-1.0).
            Defaults to 0.2.
        data_base_path (str, optional): Path to the base EEG data directory. Defaults to "eeg_data".
        runs_to_include (list[int] | None, optional): Specific runs to include from the dataset.
            If None, defaults to [4, 8, 12] (motor imagery runs). Defaults to None.
        model_name (str, optional): Name for the current model run. This will be used to create
            a subdirectory under "models/" for saving model files and plots. Defaults to "unnamed_model".
        model_params_json (str | None, optional): JSON string containing additional keyword
            arguments for the EEGInceptionERPModel constructor. Defaults to None.

    Returns:
        dict: A dictionary containing key training outcomes:
            - "cv_mean_accuracy" (float): Mean accuracy across K-folds.
            - "cv_std_accuracy" (float): Standard deviation of accuracy across K-folds.
            - "final_test_accuracy" (float): Accuracy on the final hold-out test set.
            - "plot_path" (str): Path to the saved summary plot image.
            - "model_name" (str): The name of the model used for this run.

    Raises:
        ValueError: If `subjects_to_use` is invalid type, or if `test_split_ratio` is not in [0,1].
        FileNotFoundError: If `data_base_path` does not exist or is inaccessible.
        RuntimeError: If no data is loaded (e.g., `windows.size == 0`).
        # Other exceptions can be raised by underlying functions (data loading, model training).
    """
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model_params = {}
        if model_params_json:
            try:
                import json
                model_params = json.loads(model_params_json)
                print(f"Using model parameters from JSON: {model_params}")
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse model_params_json: {e}. Using default model parameters.")

        # Define the base path for saving models and plots for this run
        current_model_base_path = "models" # Base directory for all models
        # model_specific_path = os.path.join(current_model_base_path, model_name) # Path for this specific model run
        # os.makedirs(model_specific_path, exist_ok=True) # Create the directory for this model name

        if subjects_to_use is None:
            # Get all available subjects from the data loader instead of defaulting to 1-10
            data_loader_temp = BCIDataLoader(data_base_path)
            subjects_list = data_loader_temp.get_available_subjects()
            print(f"No subjects specified, using all available subjects: {len(subjects_list)} subjects found")
        elif isinstance(subjects_to_use, str) and subjects_to_use.lower() == 'all':
            # Get all available subjects from the data loader
            data_loader_temp = BCIDataLoader(data_base_path)
            subjects_list = data_loader_temp.get_available_subjects()
            print(f"Using all available subjects: {len(subjects_list)} subjects found")
        elif isinstance(subjects_to_use, list):
            subjects_list = subjects_to_use
            print(f"Using specified subjects: {subjects_list}")
        else:
            raise ValueError("subjects_to_use must be a list of integers, 'all', or None.")

        if runs_to_include is None:
            runs_to_include = [4, 8, 12] # Default motor imagery runs

        # Load data
        data_loader = BCIDataLoader(
            data_path=data_base_path, # Use data_base_path parameter
            subjects=subjects_list,
            runs=runs_to_include  # Use runs_to_include parameter
        )
        
        windows, labels, subject_ids = data_loader.load_all_subjects()
        if windows.size == 0:
            raise RuntimeError("No data loaded from BCIDataLoader. Cannot proceed with training.")
        
        # Normalize the loaded data using the universal normalizer
        from src.data.data_normalizer import ImprovedEEGNormalizer
        
        # FIXADO: Usar exatamente o mesmo normalizador do notebook
        normalizer = ImprovedEEGNormalizer(
            method='robust_zscore',    # Exato do notebook
            scope='channel',           # Exato do notebook
            outlier_threshold=3.0      # Exato do notebook
        )
        windows = normalizer.fit_transform(windows)
        
        # Debug: save normalized training data to disk for inspection
        from pathlib import Path
        import pandas as pd
        debug_folder = Path("debug/normalized_training")
        debug_folder.mkdir(parents=True, exist_ok=True)
        # Flatten windows from shape (n_windows, n_channels, n_timepoints) to (n_windows, n_channels*n_timepoints)
        windows_flat = windows.reshape(windows.shape[0], -1)
        df = pd.DataFrame(windows_flat)
        df.to_csv(debug_folder / "normalized_windows_training.csv", index=False)
        
        # Debug: save normalized training data to disk for inspection
        import os, numpy as np
        debug_folder = os.path.join("debug", "normalized_training")
        os.makedirs(debug_folder, exist_ok=True)
        np.save(os.path.join(debug_folder, "normalized_windows.npy"), windows)
        
        print(f"Data loaded. Shapes: Windows-{windows.shape}, Labels-{labels.shape}, Subject IDs-{subject_ids.shape}")
        print(f"Unique labels: {np.unique(labels)}, Counts: {np.bincount(labels)}")
        print(f"Number of unique subjects in loaded data: {len(np.unique(subject_ids))}")


        # Hold out test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            windows, labels, test_size=test_split_ratio, random_state=42, stratify=labels # Use test_split_ratio
        )
        
        # K-fold cross-validation
        kfold = KFold(n_splits=num_k_folds, shuffle=True, random_state=42) # Use num_k_folds parameter
        
        # Track results across folds
        fold_results = []
        all_training_histories = []
        
        print(f"\nStarting {num_k_folds}-fold cross-validation for model: {model_name}...") # Use num_k_folds parameter
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val), 1):
            # Split data for this fold
            X_train_fold = X_train_val[train_idx]
            X_val_fold = X_train_val[val_idx]
            y_train_fold = y_train_val[train_idx]
            y_val_fold = y_train_val[val_idx]
            
            # Train model for this fold
            best_val_acc, training_history = train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold, device,
                num_epochs=num_epochs_per_fold, 
                learning_rate=learning_rate, 
                early_stopping_patience=early_stopping_patience, 
                batch_size=batch_size, 
                model_base_path=current_model_base_path, 
                model_name=model_name, 
                model_params=model_params
            )
            
            fold_results.append(best_val_acc)
            all_training_histories.append(training_history)
        
        # Calculate cross-validation statistics
        cv_mean = np.mean(fold_results)
        cv_std = np.std(fold_results)
        
        print(f"\n{'='*50}")
        print(f"K-FOLD CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Individual fold accuracies:")
        for i, acc in enumerate(fold_results, 1):
            print(f"  Fold {i}: {acc:.4f}")
        print(f"\nMean CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Model Type: EEGInceptionERP") # Explicitly state model type
        print(f"{'='*50}")
        
        # Train final model on all training data and evaluate on test set
        print(f"\\nTraining final model on all training data ({model_name})...")
        final_best_acc, final_history = train_single_fold(
            X_train_val, y_train_val, X_test, y_test, "final", device,
            num_epochs=num_epochs_per_fold, 
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience, 
            batch_size=batch_size,
            model_base_path=current_model_base_path, 
            model_name=model_name, 
            model_params=model_params
        )
        
        print(f"\\nFinal test accuracy for {model_name} (EEGInceptionERP): {final_best_acc:.4f}")
        
        # Generate comprehensive plots
        print(f"\\nGenerating training plots for {model_name}...")
        # plots_base_dir = "plots" # Old base directory for plots
        model_specific_plots_path = os.path.join(current_model_base_path, model_name) # Save plots inside the model's folder
        os.makedirs(model_specific_plots_path, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 12)) # Increased figure size for better layout with legend
        
        # Subplot 1: CV accuracy distribution
        plt.subplot(2, 3, 1)
        plt.bar(range(1, num_k_folds + 1), fold_results, alpha=0.7, color='skyblue', edgecolor='navy') # Use num_k_folds
        plt.axhline(y=cv_mean, color='red', linestyle='--', label=f'Mean: {cv_mean:.3f}')
        plt.fill_between(range(0, num_k_folds + 2), cv_mean - cv_std, cv_mean + cv_std, # Use num_k_folds
                         alpha=0.2, color='red', label=f'±1 STD: {cv_std:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Validation Accuracy')
        plt.title('K-Fold Cross-Validation Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Average learning curves across folds
        plt.subplot(2, 3, 2)
        if all_training_histories:
            max_epochs = max(len(hist['val_accuracies']) for hist in all_training_histories if hist['val_accuracies']) if any(hist['val_accuracies'] for hist in all_training_histories) else 0
            
            val_acc_mean = []
            val_acc_std = []
            if max_epochs > 0: 
                for epoch in range(max_epochs):
                    epoch_accs = []
                    for hist in all_training_histories:
                        if epoch < len(hist['val_accuracies']):
                            epoch_accs.append(hist['val_accuracies'][epoch])
                    if epoch_accs: 
                        val_acc_mean.append(np.mean(epoch_accs))
                        val_acc_std.append(np.std(epoch_accs))
            
            if val_acc_mean: 
                epochs_range = range(1, len(val_acc_mean) + 1)
                plt.plot(epochs_range, val_acc_mean, 'b-', linewidth=2, label='Mean Val Accuracy')
                plt.fill_between(epochs_range, 
                                 np.array(val_acc_mean) - np.array(val_acc_std),
                                 np.array(val_acc_mean) + np.array(val_acc_std),
                                 alpha=0.3, color='blue')
            else: 
                plt.text(0.5, 0.5, "No data for learning curves", horizontalalignment='center', verticalalignment='center')

        else: 
            plt.text(0.5, 0.5, "No training history available", horizontalalignment='center', verticalalignment='center')

        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Average Learning Curve Across Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Final model training curve
        plt.subplot(2, 3, 3)
        final_epochs = range(1, len(final_history['train_accuracies']) + 1)
        plt.plot(final_epochs, final_history['train_accuracies'], 'b-', label='Training')
        plt.plot(final_epochs, final_history['val_accuracies'], 'r-', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Final Model Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Loss curves for final model
        plt.subplot(2, 3, 4)
        plt.plot(final_epochs, final_history['train_losses'], 'b-', label='Training Loss')
        plt.plot(final_epochs, final_history['val_losses'], 'r-', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Final Model Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Individual fold learning curves
        ax5 = plt.subplot(2, 3, 5)
        for i, hist in enumerate(all_training_histories):
            fold_epochs = range(1, len(hist['val_accuracies']) + 1)
            ax5.plot(fold_epochs, hist['val_accuracies'], label=f'Fold {i+1} Val Acc') # Added plotting logic
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Validation Accuracy')
        ax5.set_title('Individual Fold Learning Curves')
        ax5.legend(fontsize='small') # Simpler legend, adjust if needed
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        # Add model_type to summary text
        summary_text = f"""
        Model Type: EEGInceptionERP
        K-Fold Cross-Validation Summary
        
        Number of Folds: {num_k_folds} 
        Mean Accuracy: {cv_mean:.4f}
        Standard Deviation: {cv_std:.4f}
        Min Accuracy: {min(fold_results):.4f}
        Max Accuracy: {max(fold_results):.4f}
        
        Final Test Accuracy: {final_best_acc:.4f}
        
        Training Parameters:
        • Batch Size: {batch_size}
        • Max Epochs: {num_epochs_per_fold}
        • Learning Rate: {learning_rate}
        • Early Stopping: {early_stopping_patience} epochs
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust rect to ensure all titles/labels fit
        
        plot_filename = os.path.join(model_specific_plots_path, 'kfold_training_results.png') # Save plot in model specific path
        plt.savefig(plot_filename)
        print(f"Training plots saved to {plot_filename}")
        # plt.show() # This was causing the crash with 'Agg' backend.
        plt.close(fig) # Explicitly close the figure to free resources.

        return {"cv_mean_accuracy": cv_mean, "cv_std_accuracy": cv_std, "final_test_accuracy": final_best_acc, "plot_path": plot_filename, "model_name": model_name}

    except Exception as e:
        import traceback
        # Save the full error traceback for debugging purposes
        error_log_folder = os.path.join("debug")
        os.makedirs(error_log_folder, exist_ok=True)
        with open(os.path.join(error_log_folder, "training_error.log"), "w") as f:
            f.write(traceback.format_exc())
        print("Something went wrong during the training. Check debug/training_error.log for details.")
        raise

if __name__ == '__main__':
    # Example usage for CLI, can be expanded with argparse
    # This part is more for direct script execution testing, CLI will call main() differently
    parser = argparse.ArgumentParser(description="Train EEG Model with K-Fold CV.")
    parser.add_argument('--subjects', type=int, nargs='+', default=None, help='List of subject IDs to use for training (e.g., 1 2 3).')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per fold.')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio for hold-out test set.')
    parser.add_argument('--data_path', type=str, default='eeg_data', help='Path to the EEG data directory.')
    parser.add_argument('--runs', type=int, nargs='+', default=None, help='List of runs to include (e.g., 4 8 12). Defaults to motor imagery runs.')
    parser.add_argument('--model_name', type=str, default="unnamed_bci_model", help='Name for the model run, used for saving outputs.')
    parser.add_argument('--model_params_json', type=str, default=None, help='JSON string of additional model parameters.')

    args = parser.parse_args()

    results = main(
        subjects_to_use=args.subjects,
        num_epochs_per_fold=args.epochs,
        num_k_folds=args.k_folds,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        batch_size=args.batch_size,
        test_split_ratio=args.test_split,
        data_base_path=args.data_path,
        runs_to_include=args.runs,
        model_name=args.model_name,
        model_params_json=args.model_params_json
    )
    print("\\nTraining Run Summary:")
    print(results)
    print(results)
