"""
BCI Application Launcher

This script serves as the main entry point for the BCI application.
It handles initialization, dependency checks, and launches the GUI
or other operational modes via command-line arguments.
"""

import sys # Added: for sys.exit and path manipulation
import os
import logging # Added: for logging
import warnings # Added: for suppressing warnings
from pathlib import Path
import pkg_resources
import argparse # Added: for command-line argument parsing
from src.UI.main_gui import start_gui # Added: Import the GUI start function

# Suppress expected warnings from ML libraries
warnings.filterwarnings("ignore", category=UserWarning, module="braindecode.models.base")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
warnings.filterwarnings("ignore", message="LogSoftmax final layer will be removed*")
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths*")

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
        'scipy': '1.5.0',
        'braindecode': '0.8.0'
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
    
    # Configure warning filters for cleaner output
    logging.captureWarnings(True)
    
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
    parser = argparse.ArgumentParser(description="BCI Application Launcher")
    parser.add_argument(
        '--mode',
        type=str,
        default='gui',
        choices=['gui', 'train'], # Add other modes like 'preprocess', 'evaluate' later
        help="Operation mode: 'gui' to launch the GUI to run model training."
    )
    # Add other arguments for specific modes as needed, e.g., config file for training
    # parser.add_argument('--config', type=str, help="Path to configuration file for training/evaluation")

    args = parser.parse_args()

    # Check Python version
    if sys.version_info < (3, 8):
        logging.error("Python 3.8 or higher is required.") # Corrected: Use logging
        sys.exit(1) # Corrected: Use sys.exit
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        logging.error(f"Missing dependencies: {', '.join(missing)}. Please install them.") # Corrected: Use logging
        sys.exit(1) # Corrected: Use sys.exit
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting BCI application in {args.mode} mode")

    if args.mode == 'gui':
        logger.info("Launching GUI...")
        start_gui()    
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
