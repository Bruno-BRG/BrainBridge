import sys
from PyQt5.QtWidgets import QApplication
from bci.config import ensure_folders_exist
from bci.database import DatabaseManager
from bci.ui.main_window import MainWindow

def main():
    # Garante que todas as pastas existem
    ensure_folders_exist()

    app = QApplication(sys.argv)
    db = DatabaseManager()
    window = MainWindow(db)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
