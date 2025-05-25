import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PyQt5.QtCore import QThread, pyqtSignal
from src.model.eeg_inception_erp import EEGModel
from src.data.data_loader import BCIDataset


class TrainingThread(QThread):
    """Thread for training models in background"""
    progress_updated = pyqtSignal(int)
    epoch_completed = pyqtSignal(int, float, float, float, float)  # epoch, train_loss, train_acc, val_loss, val_acc
    training_completed = pyqtSignal(object, str)  # model, save_path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, X_train, y_train, X_val, y_val, model_config, training_config, save_path):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_config = model_config
        self.training_config = training_config
        self.save_path = save_path
        
    def run(self):
        try:
            # Initialize model
            model = EEGModel(**self.model_config)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Prepare data
            train_dataset = BCIDataset(self.X_train, self.y_train)
            val_dataset = BCIDataset(self.X_val, self.y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
            
            # Initialize training components
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.training_config['learning_rate'])
            
            num_epochs = self.training_config['num_epochs']
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, labels in train_loader:
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
                
                # Emit epoch results
                self.epoch_completed.emit(epoch + 1, train_loss, train_acc, val_loss, val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.training_config['patience']:
                        break
                
                # Update progress
                progress = int((epoch + 1) / num_epochs * 100)
                self.progress_updated.emit(progress)
            
            self.training_completed.emit(model, self.save_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Training error: {str(e)}")
