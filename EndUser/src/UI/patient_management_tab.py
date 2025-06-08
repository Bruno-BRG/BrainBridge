"""
Class:   PatientManagementTab
Purpose: Manages patient registration and selection for BCI rehabilitation.
         Allows medical professionals to register new patients with clinical data
         and select existing patients for EEG recording and model training.
Author:  Copilot
Created: 2025-06-07
Modified: 2025-06-07
"""
import os
import sys
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QLineEdit, QPushButton, QMessageBox, QGroupBox,
    QFormLayout, QRadioButton, QButtonGroup, QShortcut,
    QComboBox, QSpinBox, QTextEdit, QDateEdit, QListWidget,
    QListWidgetItem, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QKeySequence

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.data.patient_data_manager import PatientDataManager
except ImportError:
    PatientDataManager = None


class PatientManagementTab(QWidget):
    patient_selected = pyqtSignal(str, dict)  # Signal when patient is selected
    
    def __init__(self, parent_main_window):
        super().__init__()
        self.main_window = parent_main_window
        self.patients_data_file = os.path.join(project_root, "patient_data", "patients_registry.json")
        self.current_patient_data = None
        
        # Ensure patient data directory exists
        os.makedirs(os.path.dirname(self.patients_data_file), exist_ok=True)
        
        self.setup_ui()
        self.load_patients_list()
        
    def setup_ui(self):
        """Set up patient management interface."""
        # Apply consistent styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 8px 2px 2px 2px;
                padding-top: 15px;
                background-color: #f8f8f8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: #f8f8f8;
                color: #2d2d2d;
            }
            QLabel {
                font-weight: 500;
                font-size: 11px;
                color: #2d2d2d;
                padding: 3px 5px;
                background-color: transparent;
                min-height: 16px;
            }
            QLineEdit, QComboBox, QSpinBox, QDateEdit, QTextEdit {
                border: 2px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 11px;
                background-color: #ffffff;
                color: #2d2d2d;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDateEdit:focus, QTextEdit:focus {
                border: 2px solid #0078d4;
                background-color: #ffffff;
                outline: none;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QRadioButton {
                font-weight: normal;
                font-size: 11px;
                color: #2d2d2d;
                spacing: 8px;
                padding: 3px;
            }
            QListWidget {
                border: 2px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                font-size: 11px;
                alternate-background-color: #f5f5f5;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Patient selection
        left_panel = self.create_patient_selection_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Patient management (register/view)
        self.right_panel = QStackedWidget()
        self.setup_register_form()
        self.setup_patient_details_view()
        splitter.addWidget(self.right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)
        
    def create_patient_selection_panel(self):
        """Create the left panel for patient selection."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Patient list group
        patients_group = QGroupBox("Registered Patients")
        patients_layout = QVBoxLayout()
        
        self.patients_list = QListWidget()
        self.patients_list.itemClicked.connect(self.on_patient_selected)
        patients_layout.addWidget(self.patients_list)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_new_patient = QPushButton("New Patient")
        self.btn_new_patient.clicked.connect(self.show_register_form)
        
        self.btn_select_patient = QPushButton("Select Patient")
        self.btn_select_patient.clicked.connect(self.select_current_patient)
        self.btn_select_patient.setEnabled(False)
        
        btn_layout.addWidget(self.btn_new_patient)
        btn_layout.addWidget(self.btn_select_patient)
        patients_layout.addLayout(btn_layout)
        
        patients_group.setLayout(patients_layout)
        layout.addWidget(patients_group)
        
        # Current patient info
        current_group = QGroupBox("Current Patient")
        current_layout = QVBoxLayout()
        
        self.current_patient_label = QLabel("No patient selected")
        self.current_patient_label.setStyleSheet("color: #666; font-style: italic;")
        current_layout.addWidget(self.current_patient_label)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        layout.addStretch()
        return panel
        
    def setup_register_form(self):
        """Setup patient registration form."""
        register_widget = QWidget()
        layout = QVBoxLayout(register_widget)
        
        # Patient Information Group
        patient_info_group = QGroupBox("Patient Information")
        patient_info_layout = QFormLayout()
        
        # Basic info
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("e.g., PAT001, JOHN_001")
        patient_info_layout.addRow(QLabel("Patient ID:"), self.patient_id_input)
        
        self.patient_name_input = QLineEdit()
        self.patient_name_input.setPlaceholderText("Full name")
        patient_info_layout.addRow(QLabel("Full Name:"), self.patient_name_input)
        
        self.patient_age_input = QSpinBox()
        self.patient_age_input.setRange(1, 120)
        self.patient_age_input.setValue(30)
        patient_info_layout.addRow(QLabel("Age:"), self.patient_age_input)
        
        self.patient_gender_input = QComboBox()
        self.patient_gender_input.addItems(["Male", "Female", "Other", "Prefer not to say"])
        patient_info_layout.addRow(QLabel("Gender:"), self.patient_gender_input)
        
        patient_info_group.setLayout(patient_info_layout)
        layout.addWidget(patient_info_group)
        
        # Clinical Information Group
        clinical_group = QGroupBox("Clinical Information")
        clinical_layout = QFormLayout()
        
        # Affected hand
        self.affected_hand_input = QComboBox()
        self.affected_hand_input.addItems(["Left Hand", "Right Hand"])
        clinical_layout.addRow(QLabel("Affected Hand:"), self.affected_hand_input)
        
        # Time since onset
        self.onset_time_input = QLineEdit()
        self.onset_time_input.setPlaceholderText("e.g., 6 months, 2 years")
        clinical_layout.addRow(QLabel("Time Since Onset:"), self.onset_time_input)
        
        clinical_group.setLayout(clinical_layout)
        layout.addWidget(clinical_group)  
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        
        self.btn_save_patient = QPushButton("Register Patient")
        self.btn_save_patient.clicked.connect(self.register_patient)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.show_patient_list)
        
        action_layout.addWidget(self.btn_save_patient)
        action_layout.addWidget(self.btn_cancel)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        layout.addStretch()
        self.right_panel.addWidget(register_widget)
        
    def setup_patient_details_view(self):
        """Setup patient details view."""
        details_widget = QWidget()
        layout = QVBoxLayout(details_widget)
        
        # Patient details display (will be populated when patient is selected)
        self.patient_details_group = QGroupBox("Patient Details")
        self.patient_details_layout = QFormLayout()
        
        # Placeholder labels
        self.detail_id_label = QLabel()
        self.detail_name_label = QLabel()
        self.detail_age_label = QLabel()
        self.detail_gender_label = QLabel()
        self.detail_affected_hand_label = QLabel()
        self.detail_onset_time_label = QLabel()
        
        self.patient_details_layout.addRow(QLabel("Patient ID:"), self.detail_id_label)
        self.patient_details_layout.addRow(QLabel("Name:"), self.detail_name_label)
        self.patient_details_layout.addRow(QLabel("Age:"), self.detail_age_label)
        self.patient_details_layout.addRow(QLabel("Gender:"), self.detail_gender_label)
        self.patient_details_layout.addRow(QLabel("Affected Hand:"), self.detail_affected_hand_label)
        self.patient_details_layout.addRow(QLabel("Time Since Onset:"), self.detail_onset_time_label)
        
        self.patient_details_group.setLayout(self.patient_details_layout)
        layout.addWidget(self.patient_details_group)
        
        # Model history group (placeholder for future implementation)
        history_group = QGroupBox("Model Training History")
        history_layout = QVBoxLayout()
        
        self.history_label = QLabel("No training history available yet.")
        self.history_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        history_layout.addWidget(self.history_label)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        layout.addStretch()
        self.right_panel.addWidget(details_widget)
        
    def load_patients_list(self):
        """Load patients from file and populate the list."""
        self.patients_list.clear()
        
        if not os.path.exists(self.patients_data_file):
            return
            
        try:
            with open(self.patients_data_file, 'r') as f:
                patients_data = json.load(f)
                
            for patient_id, patient_info in patients_data.items():
                item_text = f"{patient_id} - {patient_info.get('name', 'Unknown')}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, patient_id)
                self.patients_list.addItem(item)
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.show_error_message(f"Error loading patients: {str(e)}")
            
    def on_patient_selected(self, item):
        """Handle patient selection from list."""
        patient_id = item.data(Qt.UserRole)
        self.load_patient_details(patient_id)
        self.btn_select_patient.setEnabled(True)
        
    def load_patient_details(self, patient_id):
        """Load and display patient details."""
        try:
            with open(self.patients_data_file, 'r') as f:
                patients_data = json.load(f)
                
            if patient_id in patients_data:
                patient_info = patients_data[patient_id]
                self.current_patient_data = patient_info
                
                # Update detail labels
                self.detail_id_label.setText(patient_id)
                self.detail_name_label.setText(patient_info.get('name', 'N/A'))
                self.detail_age_label.setText(str(patient_info.get('age', 'N/A')))
                self.detail_gender_label.setText(patient_info.get('gender', 'N/A'))
                self.detail_affected_hand_label.setText(patient_info.get('affected_hand', 'N/A'))
                self.detail_onset_time_label.setText(patient_info.get('onset_time', 'N/A'))
                
                # Show details view
                self.right_panel.setCurrentIndex(1)
                
        except Exception as e:
            self.show_error_message(f"Error loading patient details: {str(e)}")
            
    def register_patient(self):
        """Register a new patient."""
        # Validate inputs
        patient_id = self.patient_id_input.text().strip()
        patient_name = self.patient_name_input.text().strip()
        
        if not patient_id or not patient_name:
            self.show_error_message("Patient ID and Name are required.")
            return
            
        # Check if patient ID already exists
        if os.path.exists(self.patients_data_file):
            try:
                with open(self.patients_data_file, 'r') as f:
                    patients_data = json.load(f)
                    
                if patient_id in patients_data:
                    self.show_error_message(f"Patient ID '{patient_id}' already exists.")
                    return
                    
            except json.JSONDecodeError:
                patients_data = {}
        else:
            patients_data = {}
            
        # Create patient data
        patient_info = {
            'name': patient_name,
            'age': self.patient_age_input.value(),
            'gender': self.patient_gender_input.currentText(),
            'affected_hand': self.affected_hand_input.currentText(),
            'onset_time': self.onset_time_input.text().strip(),
            'registration_date': datetime.now().isoformat(),
            'model_history': []
        }
        
        # Save patient data
        patients_data[patient_id] = patient_info
        
        try:
            with open(self.patients_data_file, 'w') as f:
                json.dump(patients_data, f, indent=2)
                
            self.show_success_message(f"Patient '{patient_name}' registered successfully!")
            self.clear_register_form()
            self.load_patients_list()
            self.show_patient_list()
            
        except Exception as e:
            self.show_error_message(f"Error saving patient data: {str(e)}")
            
    def clear_register_form(self):
        """Clear registration form."""
        self.patient_id_input.clear()
        self.patient_name_input.clear()
        self.patient_age_input.setValue(30)
        self.patient_gender_input.setCurrentIndex(0)
        self.affected_hand_input.setCurrentIndex(0)
        self.onset_time_input.clear()
        
    def show_register_form(self):
        """Show patient registration form."""
        self.right_panel.setCurrentIndex(0)
        self.clear_register_form()
        
    def show_patient_list(self):
        """Show patient details view."""
        self.right_panel.setCurrentIndex(1)
        
    def select_current_patient(self):
        """Select the current patient for the session."""
        current_item = self.patients_list.currentItem()
        if current_item:
            patient_id = current_item.data(Qt.UserRole)
            if self.current_patient_data:
                # Update main window
                self.main_window.current_patient_id = patient_id
                self.main_window.current_patient_data = self.current_patient_data
                
                # Update current patient label
                self.current_patient_label.setText(f"Active: {patient_id} - {self.current_patient_data['name']}")
                self.current_patient_label.setStyleSheet("color: #0078d4; font-weight: bold;")
                
                # Emit signal
                self.patient_selected.emit(patient_id, self.current_patient_data)
                
                self.show_success_message(f"Patient '{patient_id}' selected for current session.")
                
    def show_success_message(self, message, title="Success"):
        """Show success message."""
        QMessageBox.information(self, title, message)
        
    def show_error_message(self, message, title="Error"):
        """Show error message."""
        QMessageBox.warning(self, title, message)


# For backward compatibility with main_gui.py
AuthTab = PatientManagementTab
