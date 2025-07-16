class BCIMainWindow(QMainWindow):
    """Janela principal da aplicação BCI"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface principal"""
        self.setWindowTitle("Sistema BCI - Interface de Controle")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central com abas
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Título
        title_label = QLabel("Sistema BCI - Interface de Controle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title_label)
        
        # Abas
        self.tabs = QTabWidget()
        
        # Aba de pacientes
        self.patient_widget = PatientRegistrationWidget(self.db_manager)
        self.tabs.addTab(self.patient_widget, "Cadastro de Pacientes")
        
        # Aba de streaming
        self.streaming_widget = StreamingWidget(self.db_manager)
        self.tabs.addTab(self.streaming_widget, "Streaming e Gravação")
        
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        
        # Barra de status
        self.statusBar().showMessage("Sistema BCI inicializado")
    
    def closeEvent(self, event):
        """Evento de fechamento da aplicação"""
        # Parar streaming se estiver rodando
        if hasattr(self.streaming_widget, 'streaming_thread') and \
           self.streaming_widget.streaming_thread is not None:
            self.streaming_widget.streaming_thread.stop_streaming()
        
        event.accept()


def main():
    """Função principal"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Estilo moderno
    
    # Aplicar tema escuro (opcional)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin: 3px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """)
    
    window = BCIMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()