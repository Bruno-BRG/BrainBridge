from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel, 
                           QLineEdit, QDateEdit, QTextEdit, QPushButton, 
                           QGroupBox, QTableWidget, QTableWidgetItem, 
                           QHeaderView, QMessageBox, QHBoxLayout, 
                           QComboBox, QSpinBox)
from PyQt5.QtCore import QDate
from typing import Optional
from ..database.database_manager import DatabaseManager

class PatientRegistrationWidget(QWidget):
    """Widget para cadastro de pacientes"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        self.load_patients()
        
    def setup_ui(self):
        """Configura a interface"""
        layout = QVBoxLayout()
        
        # Formulário de cadastro
        form_group = QGroupBox("Cadastro de Novo Paciente")
        form_layout = QGridLayout()
        
        # Campos do formulário
        form_layout.addWidget(QLabel("Nome:"), 0, 0)
        self.name_edit = QLineEdit()
        form_layout.addWidget(self.name_edit, 0, 1)
        
        form_layout.addWidget(QLabel("Idade:"), 0, 2)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setValue(30)
        form_layout.addWidget(self.age_spin, 0, 3)
        
        form_layout.addWidget(QLabel("Sexo:"), 1, 0)
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Masculino", "Feminino", "Outro"])
        form_layout.addWidget(self.sex_combo, 1, 1)
        
        form_layout.addWidget(QLabel("Mão Afetada:"), 1, 2)
        self.hand_combo = QComboBox()
        self.hand_combo.addItems(["Esquerda", "Direita", "Ambas", "Nenhuma"])
        form_layout.addWidget(self.hand_combo, 1, 3)
        
        form_layout.addWidget(QLabel("Tempo desde evento (meses):"), 2, 0)
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 1000)
        self.time_spin.setValue(0)
        form_layout.addWidget(self.time_spin, 2, 1)
        
        form_layout.addWidget(QLabel("Observações:"), 3, 0)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        form_layout.addWidget(self.notes_edit, 3, 1, 1, 3)
        
        # Botão de cadastro
        self.register_btn = QPushButton("Cadastrar Paciente")
        self.register_btn.clicked.connect(self.register_patient)
        form_layout.addWidget(self.register_btn, 4, 0, 1, 4)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Tabela de pacientes
        patients_group = QGroupBox("Pacientes Cadastrados")
        patients_layout = QVBoxLayout()
        
        self.patients_table = QTableWidget()
        self.patients_table.setColumnCount(7)
        self.patients_table.setHorizontalHeaderLabels([
            "ID", "Nome", "Idade", "Sexo", "Mão Afetada", "Tempo (meses)", "Data Cadastro"
        ])
        self.patients_table.setSelectionBehavior(QTableWidget.SelectRows)
        patients_layout.addWidget(self.patients_table)
        
        patients_group.setLayout(patients_layout)
        layout.addWidget(patients_group)
        
        self.setLayout(layout)
        
    def register_patient(self):
        """Registra um novo paciente"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Erro", "Nome é obrigatório!")
            return
            
        age = self.age_spin.value()
        sex = self.sex_combo.currentText()
        affected_hand = self.hand_combo.currentText()
        time_since_event = self.time_spin.value()
        notes = self.notes_edit.toPlainText()
        
        try:
            patient_id = self.db_manager.add_patient(
                name, age, sex, affected_hand, time_since_event, notes
            )
            
            QMessageBox.information(self, "Sucesso", 
                                  f"Paciente {name} cadastrado com ID {patient_id}")
            
            # Limpar formulário
            self.name_edit.clear()
            self.age_spin.setValue(30)
            self.sex_combo.setCurrentIndex(0)
            self.hand_combo.setCurrentIndex(0)
            self.time_spin.setValue(0)
            self.notes_edit.clear()
            
            # Recarregar tabela
            self.load_patients()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao cadastrar paciente: {e}")
    
    def load_patients(self):
        """Carrega a lista de pacientes"""
        try:
            patients = self.db_manager.get_all_patients()
            
            self.patients_table.setRowCount(len(patients))
            
            for row, patient in enumerate(patients):
                self.patients_table.setItem(row, 0, QTableWidgetItem(str(patient["id"])))
                self.patients_table.setItem(row, 1, QTableWidgetItem(patient["name"]))
                self.patients_table.setItem(row, 2, QTableWidgetItem(str(patient["age"])))
                self.patients_table.setItem(row, 3, QTableWidgetItem(patient["sex"]))
                self.patients_table.setItem(row, 4, QTableWidgetItem(patient["affected_hand"]))
                self.patients_table.setItem(row, 5, QTableWidgetItem(str(patient["time_since_event"])))
                self.patients_table.setItem(row, 6, QTableWidgetItem(patient["created_at"][:10]))
            
            self.patients_table.resizeColumnsToContents()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")
    
    def get_selected_patient(self) -> Optional[int]:
        """Retorna o ID do paciente selecionado"""
        current_row = self.patients_table.currentRow()
        if current_row >= 0:
            return int(self.patients_table.item(current_row, 0).text())
        return None
