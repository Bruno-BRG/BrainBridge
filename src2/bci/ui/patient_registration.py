from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QLineEdit, QSpinBox, QComboBox, QTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
from bci.database import DatabaseManager

class PatientRegistrationWidget(QWidget):
    """Widget para cadastro de pacientes"""
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self._build_ui()
        self._load_patients()

    def _build_ui(self):
        layout = QVBoxLayout()
        form = QGroupBox("Novo Paciente")
        grid = QGridLayout()

        grid.addWidget(QLabel("Nome:"), 0,0)
        self.name = QLineEdit()
        grid.addWidget(self.name, 0,1)

        grid.addWidget(QLabel("Idade:"), 0,2)
        self.age = QSpinBox(); self.age.setRange(0,150)
        grid.addWidget(self.age,0,3)

        grid.addWidget(QLabel("Sexo:"),1,0)
        self.sex = QComboBox(); self.sex.addItems(["Masculino","Feminino","Outro"])
        grid.addWidget(self.sex,1,1)

        grid.addWidget(QLabel("Mão Afetada:"),1,2)
        self.hand = QComboBox(); self.hand.addItems(["Esquerda","Direita","Ambas","Nenhuma"])
        grid.addWidget(self.hand,1,3)

        grid.addWidget(QLabel("Tempo (meses):"),2,0)
        self.months = QSpinBox(); self.months.setRange(0,1000)
        grid.addWidget(self.months,2,1)

        grid.addWidget(QLabel("Observações:"),3,0)
        self.notes = QTextEdit(); self.notes.setMaximumHeight(80)
        grid.addWidget(self.notes,3,1,1,3)

        self.reg_btn = QPushButton("Cadastrar")
        self.reg_btn.clicked.connect(self._register)
        grid.addWidget(self.reg_btn,4,0,1,4)

        form.setLayout(grid)
        layout.addWidget(form)

        tbl_group = QGroupBox("Pacientes Cadastrados")
        tbl_layout = QVBoxLayout()
        self.tbl = QTableWidget();
        self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels(["ID","Nome","Idade","Sexo","Mão","Tempo"])
        tbl_layout.addWidget(self.tbl)
        tbl_group.setLayout(tbl_layout)
        layout.addWidget(tbl_group)

        self.setLayout(layout)

    def _register(self):
        name = self.name.text().strip()
        if not name:
            QMessageBox.warning(self, "Erro", "Nome obrigatório")
            return
        pid = self.db.add_patient(
            name, self.age.value(), self.sex.currentText(),
            self.hand.currentText(), self.months.value(),
            self.notes.toPlainText()
        )
        QMessageBox.information(self, "Sucesso", f"Paciente ID {pid} cadastrado")
        self.name.clear(); self.age.setValue(30)
        self._load_patients()

    def _load_patients(self):
        pats = self.db.get_all_patients()
        self.tbl.setRowCount(len(pats))
        for r,p in enumerate(pats):
            self.tbl.setItem(r,0,QTableWidgetItem(str(p['id'])))
            self.tbl.setItem(r,1,QTableWidgetItem(p['name']))
            self.tbl.setItem(r,2,QTableWidgetItem(str(p['age'])))
            self.tbl.setItem(r,3,QTableWidgetItem(p['sex']))
            self.tbl.setItem(r,4,QTableWidgetItem(p['affected_hand']))
            self.tbl.setItem(r,5,QTableWidgetItem(str(p['time_since_event'])))

