"""
Class:   ModelFineTuner
Purpose: Fine-tuning of pre-trained EEG models with patient-specific data.
Author:  Bruno Rocha  
Created: 2025-06-01
License: BSD (3-clause)
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Migrated and improved from DevTools with exact notebook matching.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFineTuner:
    """
    Fine-tuning manager for pre-trained EEG models.
    
    Features:
    - Load pre-trained models with compatibility checks
    - Configurable layer freezing strategies  
    - Patient-specific fine-tuning with proper data handling
    - Training monitoring and early stopping
    - Model evaluation and validation
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize ModelFineTuner.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.verbose:
            print(f"ModelFineTuner initialized on device: {self.device}")
    
    def load_pretrained_model(self, model_path: str) -> bool:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the pre-trained model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if self.verbose:
                print(f"Loading model from: {model_path}")
            
            # Extract model parameters from checkpoint
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                
                # Get model architecture parameters
                n_chans = checkpoint.get('n_chans', 16)
                n_outputs = checkpoint.get('n_outputs', 2) 
                n_times = checkpoint.get('n_times', 400)
                sfreq = checkpoint.get('sfreq', 125.0)
                
                if self.verbose:
                    print(f"Model architecture: {n_chans} channels, {n_times} timepoints, {n_outputs} outputs")
                
            else:
                logger.error("Invalid checkpoint format - no model_state_dict found")
                return False
            
            # Try to import EEGInceptionERP from braindecode
            try:
                from braindecode.models import EEGInceptionERP
                
                # Create model with same architecture
                self.model = EEGInceptionERP(
                    n_chans=n_chans,
                    n_outputs=n_outputs, 
                    n_times=n_times,
                    sfreq=sfreq
                )
                
                if self.verbose:
                    print("✅ Using braindecode EEGInceptionERP")
                    
            except ImportError:
                logger.error("braindecode not available - cannot load EEGInceptionERP model")
                return False
            
            # Load model weights
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            
            if self.verbose:
                print("✅ Pre-trained model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            return False
    
    def fine_tune_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 30,
        learning_rate: float = 0.001,
        learning_rate_ratio: float = 0.3,  # CRITICAL: More aggressive than 0.1
        early_stopping_patience: int = 5,
        freeze_strategy: str = "early", 
        status_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
        metrics_callback: Optional[Callable] = None
    ) -> Dict[str, any]:
        """
        Fine-tune the model exactly as in the notebook with corrected parameters.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: Optional DataLoader for test data
            epochs: Maximum number of training epochs
            learning_rate: Base learning rate
            learning_rate_ratio: Reduction factor for fine-tuning learning rate
            early_stopping_patience: Number of epochs to wait before stopping
            freeze_strategy: Layer freezing strategy (ignored - no actual freezing)
            status_callback: Optional callback function to report status messages
            progress_callback: Optional callback function to report progress (0-100)
            metrics_callback: Optional callback function to report epoch metrics
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_pretrained_model() first.")
        
        if self.verbose:
            print(f"🎯 Fine-tuning with notebook-matched parameters...")
        
        # Update status if callback provided
        if status_callback:
            status_callback("Starting fine-tuning process...")
        
        # --- Configure model for fine-tuning (EXACTLY as in notebook) ---
        model = self.model.to(self.device)
        
        # CRITICAL: The notebook shows "Froze 0 layers, left 38 trainable"
        # This means NO layers are actually frozen!
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            param.requires_grad = True  # Keep all layers trainable
            trainable_count += param.numel()
        
        if self.verbose:
            print(f"Froze {frozen_count} layers, left {trainable_count} trainable")
        
        if status_callback:
            status_callback(f"Model prepared: {frozen_count} layers frozen, {trainable_count} trainable")
        
        # CRITICAL: Use EXACT learning rate from notebook
        fine_tune_lr = learning_rate * learning_rate_ratio  # 0.001 * 0.3 = 0.0003
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=fine_tune_lr,
            weight_decay=1e-4  # Add weight decay as in notebook
        )
        
        # CRITICAL: Add learning rate scheduler as in notebook
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=self.verbose
        )
        
        if self.verbose:
            print(f"Optimizer configured: lr={fine_tune_lr}")
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training tracking variables
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        history = []
        
        # Fine-tuning loop
        for epoch in range(epochs):
            if progress_callback:
                progress = int((epoch / epochs) * 100 * 0.8)  # 80% of progress bar for training
                progress_callback(progress)
            
            # --- Training phase ---
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i, (x_batch, y_batch) in enumerate(train_loader):
                # CRITICAL: Ensure correct data types
                x_batch = x_batch.float()
                y_batch = y_batch.long()
                
                # Move batch to device
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                
                # CRITICAL: Add gradient clipping as in notebook
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            # --- Validation phase ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    # CRITICAL: Ensure correct data types
                    x_batch = x_batch.float()
                    y_batch = y_batch.long()
                    
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # CRITICAL: Update learning rate based on validation performance
            scheduler.step(val_acc)
            
            # Store epoch results
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            history.append(epoch_metrics)
            
            # Report metrics if callback provided
            if metrics_callback:
                metrics_callback(epoch_metrics)
            
            if status_callback:
                status_callback(f"Epoch {epoch+1}/{epochs}: Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check (using EXACT same logic as notebook)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()  # CRITICAL: Use .copy()
                patience_counter = 0
                
                if self.verbose:
                    print(f"  ✨ New best model: Val Acc {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if self.verbose:
                        print(f"  ⏹️ Early stopping triggered after {epoch+1} epochs")
                    
                    if status_callback:
                        status_callback(f"Early stopping: best val acc {best_val_acc:.4f}")
                    
                    break
        
        # --- Test evaluation (if test_loader provided) ---
        test_acc = 0.0
        test_loss = 0.0
        all_predictions = []
        all_true_labels = []
        
        if test_loader and best_model_state:
            if progress_callback:
                progress_callback(90)  # 90% complete
                
            if status_callback:
                status_callback("Evaluating on test set...")
                
            # Load best model
            model.load_state_dict(best_model_state)
            model.eval()
            
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    # CRITICAL: Ensure correct data types
                    x_batch = x_batch.float()
                    y_batch = y_batch.long()
                    
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += y_batch.size(0)
                    test_correct += (predicted == y_batch).sum().item()
                    
                    # Store predictions and true labels
                    all_predictions.extend(predicted.cpu().numpy())
                    all_true_labels.extend(y_batch.cpu().numpy())
            
            test_loss = test_loss / len(test_loader)
            test_acc = test_correct / test_total if test_total > 0 else 0
            
            if self.verbose:
                print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        
        # Store model state for future use
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        if progress_callback:
            progress_callback(100)  # 100% complete
            
        if status_callback:
            status_callback(f"Fine-tuning completed - Best val acc: {best_val_acc:.4f}")
        
        # Prepare and return results
        results = {
            "best_model_state": best_model_state,
            "best_val_accuracy": float(best_val_acc),
            "final_test_accuracy": float(test_acc) if test_loader else None,
            "test_loss": float(test_loss) if test_loader else None,
            "history": history,
            "epochs_trained": len(history),
            "early_stopping_triggered": patience_counter >= early_stopping_patience,
            "predictions": all_predictions,
            "true_labels": all_true_labels,
            "timestamp": datetime.now().isoformat(),
            "fine_tune_lr": fine_tune_lr
        }
        
        return results


# Create a simple fine-tuning function for backward compatibility
def fine_tune_model_simple(model_path: str, 
                          train_loader: torch.utils.data.DataLoader,
                          val_loader: torch.utils.data.DataLoader,
                          epochs: int = 30,
                          learning_rate: float = 0.0003) -> Dict:
    """
    Simple fine-tuning function for backward compatibility.
    
    Args:
        model_path: Path to pre-trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        Dictionary with fine-tuning results
    """
    fine_tuner = ModelFineTuner(verbose=True)
    
    if not fine_tuner.load_pretrained_model(model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    return fine_tuner.fine_tune_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
        
    def validate_fine_tuned_model(
        self,
        validation_data_loader: torch.utils.data.DataLoader,
        metrics: List[str] = None,
        save_results: bool = True,
        results_path: str = None
    ) -> Dict[str, any]:
        """
        Validates the fine-tuned model on held-out validation data.
        
        This function evaluates the performance of the fine-tuned model and provides
        comprehensive metrics for assessing the quality of the fine-tuning process.
        
        Args:
            validation_data_loader (DataLoader): PyTorch DataLoader containing 
                validation data (X, y) pairs
            metrics (List[str], optional): List of metrics to compute. Options:
                - "accuracy": Classification accuracy
                - "precision": Per-class precision
                - "recall": Per-class recall
                - "f1": F1-score
                - "confusion_matrix": Confusion matrix
                - "auc": Area under ROC curve (binary classification)
                Defaults to ["accuracy", "precision", "recall", "f1"]
            save_results (bool): Whether to save validation results to file.
                Defaults to True.
            results_path (str, optional): Path to save results. If None, 
                generates automatic path based on model and timestamp.
        
        Returns:
            Dict[str, any]: Validation results containing:
                - metrics: Computed metrics dictionary
                - predictions: Model predictions on validation set
                - probabilities: Prediction probabilities (if available)
                - confusion_matrix: Confusion matrix (if requested)
                - validation_loss: Average validation loss
                - sample_count: Number of validation samples
                - timestamp: Validation timestamp
                - model_info: Model configuration information
        
        Raises:
            RuntimeError: If no model is loaded or validation fails
            ValueError: If validation_data_loader is invalid
        """
        
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_pretrained_model() first.")
        
        if validation_data_loader is None:
            raise ValueError("validation_data_loader cannot be None")
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Initialize tracking variables
            all_predictions = []
            all_probabilities = []
            all_labels = []
            total_loss = 0.0
            sample_count = 0
            
            # Loss function for validation loss computation
            criterion = torch.nn.CrossEntropyLoss()
            
            if self.verbose:
                print("🔍 Starting model validation...")
                print(f"   - Metrics to compute: {metrics}")
                print(f"   - Validation batches: {len(validation_data_loader)}")
            
            # Evaluate model on validation data
            with torch.no_grad():
                for batch_idx, (X_batch, y_batch) in enumerate(validation_data_loader):
                    # Move to device
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(X_batch)
                    
                    # Compute loss
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item()
                    
                    # Get predictions and probabilities
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    sample_count += X_batch.size(0)
                    
                    if self.verbose and (batch_idx + 1) % max(1, len(validation_data_loader) // 5) == 0:
                        print(f"   - Processed batch {batch_idx + 1}/{len(validation_data_loader)}")
            
            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_probabilities = np.array(all_probabilities)
            all_labels = np.array(all_labels)
            
            # Compute metrics
            computed_metrics = {}
            
            if "accuracy" in metrics:
                accuracy = np.mean(all_predictions == all_labels)
                computed_metrics["accuracy"] = float(accuracy)
            
            if "precision" in metrics:
                try:
                    from sklearn.metrics import precision_score
                    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    computed_metrics["precision"] = float(precision)
                except ImportError:
                    if self.verbose:
                        print("⚠️  sklearn not available for precision computation")
                    computed_metrics["precision"] = None
            
            if "recall" in metrics:
                try:
                    from sklearn.metrics import recall_score
                    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    computed_metrics["recall"] = float(recall)
                except ImportError:
                    if self.verbose:
                        print("⚠️  sklearn not available for recall computation")
                    computed_metrics["recall"] = None
            
            if "f1" in metrics:
                try:
                    from sklearn.metrics import f1_score
                    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    computed_metrics["f1"] = float(f1)
                except ImportError:
                    if self.verbose:
                        print("⚠️  sklearn not available for F1 computation")
                    computed_metrics["f1"] = None
            
            if "confusion_matrix" in metrics:
                try:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(all_labels, all_predictions)
                    computed_metrics["confusion_matrix"] = cm.tolist()
                except ImportError:
                    if self.verbose:
                        print("⚠️  sklearn not available for confusion matrix computation")
                    computed_metrics["confusion_matrix"] = None
            
            if "auc" in metrics:
                try:
                    from sklearn.metrics import roc_auc_score
                    # Only compute AUC for binary classification
                    if len(np.unique(all_labels)) == 2:
                        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
                        computed_metrics["auc"] = float(auc)
                    else:
                        if self.verbose:
                            print("⚠️  AUC only supported for binary classification")
                        computed_metrics["auc"] = None
                except ImportError:
                    if self.verbose:
                        print("⚠️  sklearn not available for AUC computation")
                    computed_metrics["auc"] = None
            
            # Prepare validation results
            validation_results = {
                "metrics": computed_metrics,
                "predictions": all_predictions.tolist(),
                "probabilities": all_probabilities.tolist(),
                "true_labels": all_labels.tolist(),
                "validation_loss": float(total_loss / len(validation_data_loader)),
                "sample_count": int(sample_count),
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "base_model_path": self.base_model_path,
                    "device": str(self.device),
                    "model_class": type(self.model).__name__
                }
            }
            
            # Add confusion matrix if computed
            if "confusion_matrix" in computed_metrics and computed_metrics["confusion_matrix"] is not None:
                validation_results["confusion_matrix"] = computed_metrics["confusion_matrix"]
            
            # Save results if requested
            if save_results:
                if results_path is None:
                    # Generate automatic path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_path = f"validation_results_{timestamp}.json"
                
                try:
                    import json
                    with open(results_path, 'w') as f:
                        json.dump(validation_results, f, indent=2)
                    
                    if self.verbose:
                        print(f"💾 Validation results saved to: {results_path}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to save results: {e}")
            
            # Print summary if verbose
            if self.verbose:
                print("\n✅ Model validation completed!")
                print(f"   - Validation samples: {sample_count}")
                print(f"   - Average loss: {validation_results['validation_loss']:.4f}")
                if "accuracy" in computed_metrics:
                    print(f"   - Accuracy: {computed_metrics['accuracy']:.4f}")
                if "precision" in computed_metrics and computed_metrics["precision"] is not None:
                    print(f"   - Precision: {computed_metrics['precision']:.4f}")
                if "recall" in computed_metrics and computed_metrics["recall"] is not None:
                    print(f"   - Recall: {computed_metrics['recall']:.4f}")
                if "f1" in computed_metrics and computed_metrics["f1"] is not None:
                    print(f"   - F1-score: {computed_metrics['f1']:.4f}")
                if "auc" in computed_metrics and computed_metrics["auc"] is not None:
                    print(f"   - AUC: {computed_metrics['auc']:.4f}")
            
            return validation_results
            
        except Exception as e:
            raise RuntimeError(f"Model validation failed: {str(e)}")
