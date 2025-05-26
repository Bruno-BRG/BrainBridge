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
    "data_summary": "No data loaded yet.",
    "data_path": "eeg_data", # Store data path for reuse
    "subjects_list": None # Store subject list for reuse
}

def get_int_input(prompt, default_value):
    while True:
        try:
            value_str = input(f"{prompt} (default: {default_value}): ")
            if not value_str:
                return default_value
            return int(value_str)
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_float_input(prompt, default_value):
    while True:
        try:
            value_str = input(f"{prompt} (default: {default_value}): ")
            if not value_str:
                return default_value
            return float(value_str)
        except ValueError:
            print("Invalid input. Please enter a number.")

def handle_data_loading_cli():
    print("\n--- Data Loading Menu ---")
    global loaded_data_cache
    while True:
        print("1. Load/Process EEG Data")
        print("2. View Data Summary")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")
        if choice == '1':
            print("Calling data loading function...")
            data_path_input = input(f"Enter path to EEG data directory (default: {loaded_data_cache['data_path']}): ")
            if data_path_input:
                loaded_data_cache['data_path'] = data_path_input
            
            data_path_to_check = loaded_data_cache['data_path']
            if not os.path.isdir(data_path_to_check):
                potential_path = os.path.join(project_root, data_path_to_check)
                if os.path.isdir(potential_path):
                    loaded_data_cache['data_path'] = potential_path
                else:
                    print(f"Error: Data directory '{data_path_to_check}' not found.")
                    continue

            subjects_str = input("Enter subject IDs (comma-separated, e.g., 1,2,3 or 'all' for all available, press Enter for previously used or default): ")
            
            current_subjects_display = 'all' if loaded_data_cache["subjects_list"] is None else ",".join(map(str, loaded_data_cache["subjects_list"]))
            if not subjects_str: # User pressed Enter
                subjects_list_for_loader = loaded_data_cache["subjects_list"] # Use cached or default (None for all)
                print(f"Using previously specified subjects: {current_subjects_display if subjects_list_for_loader else 'all'}")
            elif subjects_str.lower() == 'all':
                subjects_list_for_loader = None
                loaded_data_cache["subjects_list"] = None # Update cache
            else:
                try:
                    subjects_list_for_loader = [int(s.strip()) for s in subjects_str.split(',')]
                    loaded_data_cache["subjects_list"] = subjects_list_for_loader # Update cache
                except ValueError:
                    print("Error: Invalid subject IDs. Please enter comma-separated numbers or 'all'.")
                    continue
            
            try:
                print(f"Loading data from: {loaded_data_cache['data_path']}, Subjects: {subjects_str if subjects_str else current_subjects_display}")
                loader = BCIDataLoader(data_path=loaded_data_cache['data_path'], subjects=loaded_data_cache["subjects_list"])
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
    global loaded_data_cache

    default_model_name = "cli_trained_model"
    default_num_epochs = 50
    default_k_folds = 5
    default_lr = 0.001
    default_early_stop_patience = 5
    default_batch_size = 32
    default_test_split = 0.2

    while True:
        print("1. Start Training with Custom Parameters")
        print("2. Start Training with Default Parameters")
        print("3. View Last Training Results (Not Implemented Yet)")
        print("4. Back to Main Menu")
        choice = input("Enter your choice: ")

        if choice == '1' or choice == '2':
            if loaded_data_cache["windows"] is None:
                print("No data loaded. Please load data first from the Data Management menu.")
                continue

            model_name_input = default_model_name
            num_epochs = default_num_epochs
            k_folds = default_k_folds
            lr = default_lr
            early_stop = default_early_stop_patience
            batch_s = default_batch_size
            test_s = default_test_split
            subjects_for_training_str = "all"

            if choice == '1': # Custom parameters
                print("\n--- Custom Training Parameters ---")
                model_name_input = input(f"Enter model name (default: {default_model_name}): ") or default_model_name
                num_epochs = get_int_input("Number of epochs per fold", default_num_epochs)
                k_folds = get_int_input("Number of K-folds", default_k_folds)
                lr = get_float_input("Learning rate", default_lr)
                early_stop = get_int_input("Early stopping patience", default_early_stop_patience)
                batch_s = get_int_input("Batch size", default_batch_size)
                test_s = get_float_input("Test split ratio", default_test_split)
                subjects_for_training_str = input(f"Enter subject IDs for training (e.g., 1,2,3 or 'all', default: all): ") or "all"
            else: # Default parameters
                print("\n--- Using Default Training Parameters ---")
                model_name_input = default_model_name # Ensure default name is used if not custom
            
            subjects_to_use_for_training = None
            if subjects_for_training_str.lower() != 'all':
                try:
                    subjects_to_use_for_training = [int(s.strip()) for s in subjects_for_training_str.split(',')]
                except ValueError:
                    print(f"Invalid subject IDs format: {subjects_for_training_str}. Defaulting to all loaded subjects.")
                    subjects_to_use_for_training = loaded_data_cache.get("subjects_list", None) # Fallback to loaded subjects
            else:
                subjects_to_use_for_training = loaded_data_cache.get("subjects_list", None) # Use all loaded subjects if 'all'

            print(f"\nStarting training for model: {model_name_input}...")
            try:
                results = train_main_script(
                    subjects_to_use=subjects_to_use_for_training, 
                    num_epochs_per_fold=num_epochs,
                    num_k_folds=k_folds,
                    learning_rate=lr,
                    early_stopping_patience=early_stop,
                    batch_size=batch_s,
                    test_split_ratio=test_s,
                    data_base_path=loaded_data_cache["data_path"],
                    model_name=model_name_input # Pass the model name
                )
                print("\n--- Training Completed ---")
                if isinstance(results, dict):
                    print(f"  Mean CV Accuracy: {results.get('cv_mean_accuracy', 'N/A'):.4f}")
                    print(f"  CV Std Dev: {results.get('cv_std_accuracy', 'N/A'):.4f}")
                    print(f"  Final Test Accuracy: {results.get('final_test_accuracy', 'N/A'):.4f}")
                    plot_path = results.get("plot_path")
                    model_dir = results.get("model_path")
                    if plot_path:
                        print(f"  Plots saved to: {os.path.abspath(plot_path)}")
                    if model_dir:
                        print(f"  Model artifacts saved in: {os.path.abspath(model_dir)}")
                else:
                    print(f"Training finished, but results format is unexpected: {results}")

            except Exception as e:
                print(f"An error occurred during training: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '3':
            print("Displaying last training results...")
            print("Result display placeholder. Check console output from training and files in 'plots' and root project directory for models.")
        elif choice == '4':
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
