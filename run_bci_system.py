#!/usr/bin/env python3
"""
BCI System Main Entry Point
Execute this file to start the BCI application
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import and run the main GUI
if __name__ == "__main__":
    from main_gui import main
    main()
