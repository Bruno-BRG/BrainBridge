"""
BCI Application Launcher

This script serves as the main entry point for the BCI application.
It handles initialization, dependency checks, and launches the GUI.
"""

import sys
import os
import logging
from pathlib import Path
import pkg_resources

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if all required packages are installed"""
    required = {
        'PyQt5': '5.15.0',
        'numpy': '1.19.0',
        'pandas': '1.1.0',
        'torch': '1.7.0',
        'matplotlib': '3.3.0',
        'seaborn': '0.11.0',
        'scipy': '1.5.0'
    }
    
    missing = []
    for package, min_version in required.items():
        try:
            pkg_resources.require(f"{package}>={min_version}")
        except pkg_resources.VersionConflict:
            missing.append(f"{package} (>={min_version})")
        except pkg_resources.DistributionNotFound:
            missing.append(f"{package} (>={min_version})")
    
    return missing

def setup_logging():
    """Configure logging for the application"""
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'bci_app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point for the BCI application"""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting BCI application")
    
    try:
        # Import GUI after environment checks
        from PyQt5.QtWidgets import QApplication
        from src.UI.BCIMainWindow import BCIMainWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = BCIMainWindow()
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.exception("Failed to start application")
        print(f"\nError: {str(e)}")
        print("Check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
