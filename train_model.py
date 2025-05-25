"""
Full training script for EEGInceptionERP model using real EEG data with K-fold cross-validation
"""
import torch
from torch.utils.data import DataLoader
from src.data.data_loader import BCIDataLoader
from src.model.eeg_inception_erp import EEGModel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
import matplotlib.pyplot as plt

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
K_FOLDS = 5
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

def train_single_fold(X_train, y_train, X_val, y_val, fold_num, device):
    """
    Train model for a single fold
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        fold_num: Current fold number
        device: Training device
        
    Returns:
        best_val_acc: Best validation accuracy achieved
        training_history: Dictionary with training metrics
    """
    # Create dataloaders
    train_loader = DataLoader(list(zip(X_train, y_train)), 
                            batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), 
                          batch_size=BATCH_SIZE)
    
    # Create model
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    n_classes = len(np.unique(y_train))

    model = EEGModel(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        sfreq=125
    ).to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining Fold {fold_num}...")
    for epoch in range(NUM_EPOCHS):
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
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model for this fold
            torch.save(model.state_dict(), f'eeg_inception_fold_{fold_num}.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_loader = BCIDataLoader(
        data_path="eeg_data",
        subjects=list(range(1, 11)),  # First 10 subjects
        runs=[4, 8, 12]  # Motor imagery runs
    )
    
    windows, labels, _ = data_loader.load_all_subjects()
    
    # Hold out test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        windows, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    
    # K-fold cross-validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Track results across folds
    fold_results = []
    all_training_histories = []
    
    print(f"\nStarting {K_FOLDS}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val), 1):
        # Split data for this fold
        X_train_fold = X_train_val[train_idx]
        X_val_fold = X_train_val[val_idx]
        y_train_fold = y_train_val[train_idx]
        y_val_fold = y_train_val[val_idx]
        
        # Train model for this fold
        best_val_acc, training_history = train_single_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold, device
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
    print(f"\nTraining final model on all training data...")
    final_best_acc, final_history = train_single_fold(
        X_train_val, y_train_val, X_test, y_test, "final", device
    )
    
    print(f"\nFinal test accuracy: {final_best_acc:.4f}")
    
    # Generate comprehensive plots
    print("\nGenerating training plots...")
    os.makedirs('plots', exist_ok=True)
    
    # Plot CV results summary
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: CV accuracy distribution
    plt.subplot(2, 3, 1)
    plt.bar(range(1, K_FOLDS + 1), fold_results, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(y=cv_mean, color='red', linestyle='--', label=f'Mean: {cv_mean:.3f}')
    plt.fill_between(range(0, K_FOLDS + 2), cv_mean - cv_std, cv_mean + cv_std, 
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
    plt.subplot(2, 3, 5)
    for i, hist in enumerate(all_training_histories):
        epochs = range(1, len(hist['val_accuracies']) + 1)
        plt.plot(epochs, hist['val_accuracies'], alpha=0.6, label=f'Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Individual Fold Learning Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
    K-Fold Cross-Validation Summary
    
    Number of Folds: {K_FOLDS}
    Mean Accuracy: {cv_mean:.4f}
    Standard Deviation: {cv_std:.4f}
    Min Accuracy: {min(fold_results):.4f}
    Max Accuracy: {max(fold_results):.4f}
    
    Final Test Accuracy: {final_best_acc:.4f}
    
    Training Parameters:
    • Batch Size: {BATCH_SIZE}
    • Max Epochs: {NUM_EPOCHS}
    • Learning Rate: {LEARNING_RATE}
    • Early Stopping: {EARLY_STOPPING_PATIENCE} epochs
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('plots/kfold_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("K-fold training plots saved to 'plots/' directory")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
