import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

# Add project root to Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_loader import BCIDataLoader, BCIDataset # Corrected: BCIDataset was missing from previous successful edit, but present in context
from src.model.train_model import (
    main as train_main_script,
    train_single_fold,
    validate
) # Corrected import for train_model functions

# Global variable to store loaded data
loaded_data_cache = {
    "windows": None,
    "labels": None,
    "subject_ids": None,
    "data_summary": "No data loaded yet."
}

def handle_data_loading_cli():
    print("\n--- Data Loading Menu ---")
    global loaded_data_cache # Ensure we can access and potentially modify it
    while True:
        print("1. Load/Process EEG Data")
        print("2. View Data Summary")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")
        if choice == '1':
            print("Calling data loading function...")
            data_path = input("Enter path to EEG data directory (e.g., eeg_data): ")
            if not os.path.isdir(data_path):
                # Try to prepend project root if it's a relative path from project root
                potential_path = os.path.join(project_root, data_path)
                if os.path.isdir(potential_path):
                    data_path = potential_path
                else:
                    print(f"Error: Data directory '{data_path}' not found.")
                    continue

            subjects_str = input("Enter subject IDs (comma-separated, e.g., 1,2,3 or 'all' for all available): ")
            subjects_list = None
            if subjects_str.lower() != 'all':
                try:
                    subjects_list = [int(s.strip()) for s in subjects_str.split(',')]
                except ValueError:
                    print("Error: Invalid subject IDs. Please enter comma-separated numbers.")
                    continue
            
            try:
                print(f"Loading data from: {data_path}, Subjects: {subjects_str}")
                loader = BCIDataLoader(data_path=data_path, subjects=subjects_list)
                windows, labels, subject_ids = loader.load_all_subjects()

                if windows.size == 0:
                    print("No data was loaded. Please check data path and subject IDs.")
                    loaded_data_cache["data_summary"] = "Failed to load data or no data found."
                else:
                    loaded_data_cache["windows"] = windows
                    loaded_data_cache["labels"] = labels
                    loaded_data_cache["subject_ids"] = subject_ids
                    
                    summary = (
                        f"Data loaded successfully!\n"
                        f"  Total windows: {windows.shape[0]}\n"
                        f"  Window shape: ({windows.shape[1]}, {windows.shape[2]}) (channels, timepoints)\n"
                        f"  Number of unique subjects: {len(np.unique(subject_ids))}\n"
                        f"  Class distribution: {np.bincount(labels)}"
                    )
                    loaded_data_cache["data_summary"] = summary
                    print(summary)

            except FileNotFoundError as e:
                print(f"Error: {e}")
                loaded_data_cache["data_summary"] = f"Error during loading: {e}"
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                loaded_data_cache["data_summary"] = f"An unexpected error: {e}"

        elif choice == '2':
            print("\n--- Data Summary ---")
            print(loaded_data_cache["data_summary"])
            print("--------------------")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
    print("-------------------------")

def handle_model_training_cli():
    print("\n--- Model Training Menu ---")
    global loaded_data_cache # Ensure we can access and potentially modify it

    # Check if data is loaded, not strictly necessary if train_main_script handles its own loading
    # but good for user feedback if they expect CLI loaded data to be used.
    if loaded_data_cache["windows"] is None or loaded_data_cache["labels"] is None:
        print("Warning: No data has been loaded via the CLI Data Management menu.")
        print("The training script will attempt to load its default dataset.")

    while True:
        print("1. Start Full Training (K-Fold CV + Final Model + Test Evaluation)")
        print("2. View Last Training Results (Not Implemented Yet)")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")

        if choice == '1':
            print("Configuring and starting training...")
            try:
                print("Starting full training process (as defined in train_model.py)...")
                print("This will use its own data loading (from 'eeg_data', subjects 1-10), splitting, K-Fold CV, final model training, and evaluation.")
                
                train_main_script() 
                
                print("Training process completed. Check console output and generated files (plots, models).")

            except Exception as e:
                print(f"An error occurred during training: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '2':
            print("Displaying last training results...")
            print("Result display placeholder. Check console output from training and files in 'plots' and root project directory for models.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
    print("---------------------------")

def main_menu():
    print("BCI Application CLI")
    while True:
        print("\n--- Main Menu ---")
        print("1. Data Management")
        print("2. Model Training")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            handle_data_loading_cli()
        elif choice == '2':
            handle_model_training_cli()
        elif choice == '3':
            print("Exiting CLI. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
