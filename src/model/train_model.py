"""
Full training script for EEGInceptionERP model using real EEG data with K-fold cross-validation
"""
import os
import torch
import numpy as np
import matplotlib # Import matplotlib like this
matplotlib.use('Agg') # Then set backend. Must be before importing pyplot.
import matplotlib.pyplot as plt # Then import pyplot
from torch.utils.data import DataLoader
from src.data.data_loader import BCIDataLoader
from .eeg_inception_erp import EEGModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split, KFold
import argparse # Added argparse for CLI argument parsing

# Training parameters
BATCH_SIZE = 32
# NUM_EPOCHS = 50 # Will be passed as a parameter
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
# K_FOLDS = 5 # Will be passed as a parameter
TEST_SPLIT = 0.2  # Hold out test set before K-fold

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
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

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device) # Added this line
            outputs = model(inputs) # Added this line
            loss = criterion(outputs, labels) # Added this line
            
            running_loss += loss.item() # Added this line
            _, predicted = torch.max(outputs.data, 1) # Added this line
            total += labels.size(0) # Added this line
            correct += (predicted == labels).sum().item() # Added this line
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc # Added return values

def train_single_fold(X_train, y_train, X_val, y_val, fold_num, device, num_epochs, learning_rate, early_stopping_patience, batch_size, model_base_path, model_name): # Added model_name
    """
    Train model for a single fold
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        fold_num: Current fold number
        device: Training device
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        early_stopping_patience: Patience for early stopping
        batch_size: Batch size for DataLoader
        
    Returns:
        best_val_acc: Best validation accuracy achieved
        training_history: Dictionary with training metrics
    """
    # Create dataloaders
    train_loader = DataLoader(list(zip(X_train, y_train)), 
                            batch_size=batch_size, shuffle=True) # Use batch_size parameter
    val_loader = DataLoader(list(zip(X_val, y_val)), 
                          batch_size=batch_size) # Use batch_size parameter
    
    # Create model
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    n_outputs = len(np.unique(y_train))  # Use n_outputs instead of n_classes

    model = EEGModel(
        n_chans=n_channels,
        n_outputs=n_outputs,  # Updated parameter name
        n_times=n_times,
        sfreq=125
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
            fold_model_save_path = os.path.join(model_base_path, model_name, f'eeg_inception_fold_{fold_num}.pth')
            os.makedirs(os.path.dirname(fold_model_save_path), exist_ok=True) # Ensure directory for this specific model exists
            torch.save(model.state_dict(), fold_model_save_path)
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

def main(subjects_to_use=None, num_epochs_per_fold=50, num_k_folds=5, learning_rate=0.001, early_stopping_patience=5, batch_size=32, test_split_ratio=0.2, data_base_path="eeg_data", runs_to_include=None, model_name="unnamed_model"): # Added model_name parameter
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the base path for saving models and plots for this run
    current_model_base_path = "models" # Base directory for all models
    # model_specific_path = os.path.join(current_model_base_path, model_name) # Path for this specific model run
    # os.makedirs(model_specific_path, exist_ok=True) # Create the directory for this model name

    if subjects_to_use is None:
        subjects_list = list(range(1, 11)) # Default to first 10 subjects
        print("No subjects specified, defaulting to subjects 1-10.")
    elif isinstance(subjects_to_use, str) and subjects_to_use.lower() == 'all':
        subjects_list = None # BCIDataLoader handles None as all subjects
        print("Using all available subjects.")
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
        print("No data loaded. Exiting training.")
        return
    
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
            num_epochs=num_epochs_per_fold, # Pass num_epochs_per_fold
            learning_rate=learning_rate, # Pass learning_rate
            early_stopping_patience=early_stopping_patience, # Pass early_stopping_patience
            batch_size=batch_size, # Pass batch_size
            model_base_path=current_model_base_path, # Pass the base path for models
            model_name=model_name # Pass the specific model name
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
    print(f"{'='*50}")
    
    # Train final model on all training data and evaluate on test set
    print(f"\\nTraining final model on all training data ({model_name})...")
    final_best_acc, final_history = train_single_fold(
        X_train_val, y_train_val, X_test, y_test, "final", device,
        num_epochs=num_epochs_per_fold, # Pass num_epochs_per_fold for final training as well
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience, # Can be different for final model if needed
        batch_size=batch_size,
        model_base_path=current_model_base_path, # Pass the base path
        model_name=model_name # Pass the model name
    )
    
    print(f"\\nFinal test accuracy for {model_name}: {final_best_acc:.4f}")
    
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
    max_epochs = max(len(hist['val_accuracies']) for hist in all_training_histories)
    
    # Calculate mean and std across folds
    val_acc_mean = []
    val_acc_std = []
    for epoch in range(max_epochs):
        epoch_accs = []
        for hist in all_training_histories:
            if epoch < len(hist['val_accuracies']):
                epoch_accs.append(hist['val_accuracies'][epoch])
        if epoch_accs:
            val_acc_mean.append(np.mean(epoch_accs))
            val_acc_std.append(np.std(epoch_accs))
    
    epochs_range = range(1, len(val_acc_mean) + 1)
    plt.plot(epochs_range, val_acc_mean, 'b-', linewidth=2, label='Mean Val Accuracy')
    plt.fill_between(epochs_range, 
                     np.array(val_acc_mean) - np.array(val_acc_std),
                     np.array(val_acc_mean) + np.array(val_acc_std),
                     alpha=0.3, color='blue')
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
    summary_text = f"""
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

if __name__ == '__main__':
    # Example usage for CLI, can be expanded with argparse
    # This part is more for direct script execution testing, CLI will call main() differently
    parser = argparse.ArgumentParser(description="Train EEG Model with K-Fold CV.")
    parser.add_argument("--model_name", type=str, default="cli_default_model", help="Name for the model and its output folder.")
    # Add other arguments as needed (subjects, epochs, etc.)
    args = parser.parse_args()

    main(model_name=args.model_name) # Pass the model_name
