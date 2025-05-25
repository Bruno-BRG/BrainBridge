#!/usr/bin/env python3
"""
Test script to verify the BCI GUI application opens correctly
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bci_gui_app import BCIMainWindow

def main():
    print("Starting BCI GUI Test...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("BCI GUI Test")
    
    print("Creating main window...")
    window = BCIMainWindow()
    
    # Make sure the window is visible and comes to the front
    window.show()
    window.raise_()
    window.activateWindow()
    
    print(f"Window created successfully!")
    print(f"Title: {window.windowTitle()}")
    print(f"Size: {window.size().width()} x {window.size().height()}")
    print(f"Visible: {window.isVisible()}")
    
    # Set up a timer to close the application after 5 seconds (for testing)
    # Remove this in production
    timer = QTimer()
    timer.timeout.connect(lambda: print("GUI is working! You can close this window or press Ctrl+C to exit."))
    timer.start(5000)  # 5 seconds
    
    print("GUI should now be visible. If you can see the window, the application is working correctly!")
    print("Press Ctrl+C in the terminal or close the window to exit.")
    
    # Start the event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
