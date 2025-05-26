import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.model_selection import KFold
import os

class ModelTrainer:
    """Handles model training logic and state management"""
    
    def __init__(self):
        self.current_model = None
        self.training_in_progress = False
        
    def configure_training(self, config: Dict[str, Any]) -> None:
        """Configure training parameters"""
        self.config = config
        
    def prepare_data(self, windows: np.ndarray, labels: np.ndarray, subject_ids: Optional[np.ndarray] = None) -> None:
        """Prepare data for training"""
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.subject_ids = subject_ids
        
        # Create save directory if needed
        os.makedirs(self.config.get('save_dir', 'models'), exist_ok=True)
        
    def start_kfold_training(self, callbacks: Dict[str, callable]) -> None:
        """Start k-fold cross-validation training"""
        if self.training_in_progress:
            raise RuntimeError("Training already in progress")
            
        self.training_in_progress = True
        kfold = KFold(n_splits=self.config.get('k_folds', 5), shuffle=True)
        
        try:
            callbacks.get('training_started', lambda: None)()
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.windows)):
                if not self.training_in_progress:
                    break
                    
                self._train_fold(fold + 1, train_idx, val_idx, callbacks)
        finally:
            self.training_in_progress = False
            
    def start_single_training(self, callbacks: Dict[str, callable]) -> None:
        """Start training on entire dataset"""
        if self.training_in_progress:
            raise RuntimeError("Training already in progress")
            
        self.training_in_progress = True
        train_size = int((1 - self.config.get('validation_split', 0.2)) * len(self.windows))
        indices = torch.randperm(len(self.windows))
        
        try:
            callbacks.get('training_started', lambda: None)()
            self._train_fold("final", indices[:train_size], indices[train_size:], callbacks)
        finally:
            self.training_in_progress = False
            
    def stop_training(self) -> None:
        """Stop the current training process"""
        self.training_in_progress = False
        
    def _train_fold(self, fold_num: Union[int, str], train_idx: np.ndarray, val_idx: np.ndarray, callbacks: Dict[str, callable]) -> None:
        """Train a single fold or final model"""
        try:
            # Import required modules
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            from src.model.eeg_inception_erp import EEGModel
            
            # Get data for this fold
            train_windows = self.windows[train_idx]
            train_labels = self.labels[train_idx]
            val_windows = self.windows[val_idx]
            val_labels = self.labels[val_idx]
            
            # Create datasets and data loaders
            train_dataset = TensorDataset(train_windows, train_labels)
            val_dataset = TensorDataset(val_windows, val_labels)
            
            batch_size = self.config.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            n_chans = train_windows.shape[1]
            n_times = train_windows.shape[2]
            model = EEGModel(
                n_chans=n_chans,
                n_outputs=2,  # Binary classification
                n_times=n_times,
                sfreq=125.0,  # Standard EEG sampling rate
                drop_prob=self.config.get('dropout', 0.5),
                n_filters=self.config.get('n_filters', 8)
            )
            
            # Set up training components
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))
            
            # Training loop
            num_epochs = self.config.get('num_epochs', 50)
            patience = self.config.get('patience', 5)
            best_val_acc = 0.0
            patience_counter = 0
            
            callbacks.get('fold_started', lambda f: None)(fold_num)
            
            for epoch in range(num_epochs):
                if not self.training_in_progress:
                    break
                
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                # Call epoch completed callback
                callbacks.get('epoch_completed', lambda *args: None)(epoch + 1, train_loss, train_acc, val_loss, val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    save_path = os.path.join(self.config.get('save_dir', 'models'), f'best_model_fold_{fold_num}.pth')
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Call fold completed callback
            callbacks.get('fold_completed', lambda f, acc: None)(fold_num, best_val_acc)
            
        except Exception as e:
            callbacks.get('error_occurred', lambda msg: None)(f"Error in fold {fold_num}: {str(e)}")
