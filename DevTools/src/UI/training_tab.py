"""
Classes: TrainingThread, TrainingTab
Purpose: Provides the UI tab for configuring and running model training sessions.
         TrainingThread handles the training process in a separate thread to keep the UI responsive.
Author:  Copilot (NASA-style guidelines)
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Integrates with the train_model script for backend training logic.
         IMPORTANT: Fixed parameters to match training_pipeline_openbci_v2(1).ipynb exactly.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
    QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys
import traceback

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.train_model import main as train_main_script

class TrainingThread(QThread):
    training_finished = pyqtSignal(object)
    training_error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
        # FIXADO: Parâmetros exatos do notebook
        self.EXACT_TRAINING_PARAMS = {
            "num_k_folds": 10,              # Fixo: 10 folds
            "num_epochs_per_fold": 30,       # Fixo: 30 épocas
            "batch_size": 10,               # Fixo: 10
            "early_stopping_patience": 8,    # Fixo: 8 épocas
            "learning_rate": 1e-3,          # Fixo: 0.001
            "test_split_ratio": 0.2,        # Fixo: 20% teste
            "train_subject_ids": "all"      # Fixo: todos os sujeitos
        }
        
    def run(self):
        try:
            self.log_message.emit(f"Iniciando treinamento com protocolo EXATO do notebook para o modelo: {self.model_name}")
            self.log_message.emit(f"Usando parâmetros fixos:")
            for name, value in self.EXACT_TRAINING_PARAMS.items():
                self.log_message.emit(f"  - {name}: {value}")

            # Preparação da execução
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = self
            sys.stderr = self

            # Compatibilidade com PyTorch 2.6+
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # Verificação de PyTorch
            try:
                import torch
                self.log_message.emit(f"PyTorch versão: {torch.__version__}")
            except ImportError:
                self.log_message.emit("PyTorch não encontrado.")

            # FIXADO: Execução do treinamento com parâmetros fixos do notebook
            data_path = os.path.join(project_root, "eeg_data")
            
            # CRITICAL: Add debug logging for data types
            self.log_message.emit("🔧 Debugging data types before training...")
            
            results = train_main_script(
                subjects_to_use="all",
                num_epochs_per_fold=self.EXACT_TRAINING_PARAMS["num_epochs_per_fold"],
                num_k_folds=self.EXACT_TRAINING_PARAMS["num_k_folds"],
                batch_size=self.EXACT_TRAINING_PARAMS["batch_size"],
                test_split_ratio=self.EXACT_TRAINING_PARAMS["test_split_ratio"],
                learning_rate=self.EXACT_TRAINING_PARAMS["learning_rate"],
                early_stopping_patience=self.EXACT_TRAINING_PARAMS["early_stopping_patience"],
                data_base_path=data_path,
                model_name=self.model_name
            )
            
            # Restaurar stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            self.log_message.emit("Treinamento concluído com sucesso!")
            self.training_finished.emit(results)
            
        except Exception as e:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.training_error.emit(f"Erro durante treinamento: {str(e)}\n{traceback.format_exc()}")

    def write(self, message):
        self.log_message.emit(message.strip())

    def flush(self):
        pass

class TrainingTab(QWidget):
    def __init__(self, parent_main_window):
        super().__init__()
        self.main_window = parent_main_window
        self.training_thread = None

        layout = QVBoxLayout(self)

        # FIXADO: Simplificação da interface para mostrar apenas o essencial
        info_group = QGroupBox("Informações do Treinamento Fixado")
        info_layout = QFormLayout()
        
        # Mensagem explicando o treinamento fixo
        fixed_info_label = QLabel(
            "Este treinamento usa parâmetros EXATOS do notebook para garantir reprodutibilidade.\n"
            "Todos os hiperparâmetros estão fixos e não podem ser alterados.\n\n"
            "Parâmetros do notebook:\n"
            "- K-folds: 10\n"
            "- Épocas por fold: 30\n"
            "- Batch size: 10\n"
            "- Learning rate: 0.001\n"
            "- Early stopping: 8 épocas\n"
            "- Split de teste: 20%\n"
            "- Normalização: robust_zscore por canal"
        )
        fixed_info_label.setStyleSheet("color: blue;")
        info_layout.addRow(fixed_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- Model Naming Group ---
        model_naming_group = QGroupBox("Nome do Modelo")
        model_naming_layout = QFormLayout()
        self.model_name_input = QLineEdit(self.main_window.training_params_config["model_name"])
        self.model_name_input.setPlaceholderText("Digite um nome para seu modelo")
        self.model_name_input.textChanged.connect(self.update_model_name_config)
        model_naming_layout.addRow(QLabel("Nome do Modelo:"), self.model_name_input)
        model_naming_group.setLayout(model_naming_layout)
        layout.addWidget(model_naming_group)

        # --- Training Action Group ---
        training_action_group = QGroupBox("Ações de Treinamento")
        training_action_layout = QVBoxLayout()
        self.btn_start_training = QPushButton("Iniciar Treinamento (Protocolo Exato do Notebook)")
        self.btn_start_training.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        self.btn_start_training.clicked.connect(self.start_training_action)
        training_action_layout.addWidget(self.btn_start_training)
        training_action_group.setLayout(training_action_layout)
        layout.addWidget(training_action_group)

        # --- Training Log Group ---
        training_log_group = QGroupBox("Log de Treinamento & Resultados")
        training_log_layout = QVBoxLayout()
        self.training_log_display = QTextEdit()
        self.training_log_display.setReadOnly(True)
        training_log_layout.addWidget(self.training_log_display)
        training_log_group.setLayout(training_log_layout)
        layout.addWidget(training_log_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_model_name_config(self, text):
        self.main_window.training_params_config["model_name"] = text
        self.main_window.current_model_name = text

    def start_training_action(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Treinamento em andamento", "Um processo de treinamento já está em execução.")
            return

        model_name = self.model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Nome inválido", "Por favor, forneça um nome para o modelo.")
            return

        self.main_window.current_model_name = model_name

        self.training_log_display.clear()
        self.training_log_display.append(f"Preparando para treinar modelo: {model_name}")
        self.training_log_display.append("Usando o protocolo EXATO do notebook training_pipeline_openbci_v2(1).ipynb")
        self.training_log_display.append("============================================================")

        # FIXADO: Criação do thread com apenas o nome do modelo
        self.training_thread = TrainingThread(model_name)
        self.training_thread.training_finished.connect(self.training_finished)
        self.training_thread.training_error.connect(self.training_error)
        self.training_thread.log_message.connect(self.append_log_message)
        
        self.btn_start_training.setEnabled(False)
        self.btn_start_training.setText("Treinamento em andamento...")
        self.training_thread.start()

    def append_log_message(self, message):
        self.training_log_display.append(message)
        # Auto-scroll para baixo
        scrollbar = self.training_log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def training_finished(self, results):
        self.btn_start_training.setEnabled(True)
        self.btn_start_training.setText("Iniciar Treinamento (Protocolo Exato do Notebook)")
        
        # Exibir resultados
        self.training_log_display.append("\n============================================================")
        self.training_log_display.append("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        self.training_log_display.append("============================================================")
        self.training_log_display.append(f"Acurácia média CV: {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}")
        self.training_log_display.append(f"Acurácia final de teste: {results['final_test_accuracy']:.4f}")
        self.training_log_display.append(f"Plot salvo em: {results['plot_path']}")
        
    def training_error(self, error_message):
        self.btn_start_training.setEnabled(True)
        self.btn_start_training.setText("Iniciar Treinamento (Protocolo Exato do Notebook)")
        
        self.training_log_display.append("\n============================================================")
        self.training_log_display.append("❌ ERRO DE TREINAMENTO")
        self.training_log_display.append("============================================================")
        self.training_log_display.append(error_message)
        
        QMessageBox.critical(self, "Erro de Treinamento", 
                            "Ocorreu um erro durante o treinamento. Verifique o log para detalhes.")
        current_params = {}
        if self.main_window.training_params_config["use_default_params"]:
            # Use default values (already in training_params_config)
            current_params = {
                "epochs": self.main_window.training_params_config["epochs"],
                "k_folds": self.main_window.training_params_config["k_folds"],
                "learning_rate": self.main_window.training_params_config["learning_rate"],
                "early_stopping_patience": self.main_window.training_params_config["early_stopping_patience"],
                "batch_size": self.main_window.training_params_config["batch_size"],
                "test_split_size": self.main_window.training_params_config["test_split_size"],
                "train_subject_ids": self.main_window.training_params_config["train_subject_ids"]
            }
        else:
            # Use custom values from input fields
            try:
                current_params["epochs"] = self.main_window.custom_param_inputs["epochs"].value()
                current_params["k_folds"] = self.main_window.custom_param_inputs["k_folds"].value()
                current_params["learning_rate"] = self.main_window.custom_param_inputs["learning_rate"].value()
                current_params["early_stopping_patience"] = self.main_window.custom_param_inputs["early_stopping_patience"].value()
                current_params["batch_size"] = self.main_window.custom_param_inputs["batch_size"].value() # Added
                current_params["test_split_size"] = self.main_window.custom_param_inputs["test_split_size"].value() # Added
                current_params["train_subject_ids"] = self.main_window.custom_param_inputs["train_subject_ids"].text() # Added
                
                # Update the main config as well with all custom values
                self.main_window.training_params_config.update(current_params)
            except Exception as e:
                self.training_log_display.append(f"Error reading custom parameters: {e}")
                return

        model_name = self.model_name_input.text().strip()
        if not model_name:
            model_name = "unnamed_model"
            self.model_name_input.setText(model_name) # Update UI if empty
        self.main_window.current_model_name = model_name # Update main window's model name

        self.training_log_display.clear()
        self.training_log_display.append(f"Preparing to train model: {model_name}")
        self.training_log_display.append(f"Parameters: {current_params}")

        self.training_thread = TrainingThread(current_params, self.main_window.data_cache, model_name)
        self.training_thread.log_message.connect(self.append_log_message)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_error.connect(self.on_training_error)
        
        self.btn_start_training.setEnabled(False)
        self.training_thread.start()

    def append_log_message(self, message):
        self.training_log_display.append(message)

    def on_training_finished(self, results):
        self.training_log_display.append("\n--- Training Complete ---")
        if results:
            self.training_log_display.append("Results:")
            # Assuming results is a dictionary or an object with a __str__ method
            # For more detailed display, you might need to format `results`
            if isinstance(results, dict):
                for key, value in results.items():
                    self.training_log_display.append(f"  {key}: {value}")
            else:
                self.training_log_display.append(str(results))
        else:
            self.training_log_display.append("Training finished, but no results were returned.")
        self.btn_start_training.setEnabled(True)
        QMessageBox.information(self, "Training Complete", f"Training for model '{self.main_window.current_model_name}' finished successfully.")


    def on_training_error(self, error_message):
        self.training_log_display.append(f"--- TRAINING ERROR ---")
        self.training_log_display.append(error_message)
        self.btn_start_training.setEnabled(True)
        QMessageBox.critical(self, "Training Error", f"An error occurred during training for model '{self.main_window.current_model_name}'.\nCheck logs for details.")


