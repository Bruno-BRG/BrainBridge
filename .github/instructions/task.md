# Project Tasks

## Backlog
- [ ] integrate-pylsl-openbci – _Read EEG data stream from OpenBCI via pylsl_

## In Progress
- [ ] implement-gui – _Create a GUI with all features currently available in the CLI_
  - [x] 2025-05-26 gui-main-layout – _Design and implement the main layout with navigation for data and training sections._
  - [x] 2025-05-26 gui-data-load-path – _Allow user to select EEG data directory via a file dialog._
  - [x] 2025-05-26 gui-data-subject-input – _Allow user to input subject IDs (e.g., "1,2,3" or "all")._
  - [x] 2025-05-26 gui-data-load-action – _Implement button to trigger data loading and processing._
  - [x] 2025-05-26 gui-data-summary-display – _Display data summary (total windows, shape, subjects, class distribution) in the GUI._
  - [x] 2025-05-26 gui-eeg-data-plot – _Display EEG data samples with navigation (previous/next)._
  - [x] 2025-05-26 gui-training-param-toggle – _Allow user to switch between default and custom training parameters._
  - [x] 2025-05-26 gui-training-custom-params – _Create input fields for custom parameters (epochs, k-folds, learning rate, early stopping, batch size, test split)._
  - [x] 2025-05-27 gui-pylsl-integration – _Integrate PyLSL for real-time EEG streaming with simple static visualization._
  - [x] refactor-gui-data-tab-oop – _Refactor Data Management tab into its own class._
  - [x] refactor-gui-training-tab-oop – _Refactor Model Training tab into its own class._
  - [x] refactor-gui-pylsl-tab-oop – _Refactor OpenBCI Live (PyLSL) tab into its own class._
  - [x] refactor-training-thread-oop – _Move TrainingThread class to its own file._
  - [ ] gui-training-subject-input – _Allow user to specify subject IDs for training, defaulting to loaded subjects or a predefined set._
  - [ ] gui-training-start-action – _Implement button to start the model training process._
  - [ ] gui-training-progress-display – _Display training progress and results in the GUI (e.g., logs, metrics, plots)._
  - [x] 2025-05-26 gui-exit-functionality – _Implement an exit or quit button/menu option._
- [ ] refactor-model-oop – _Refactor model components to adhere to OOP principles and NASA-grade standards._
  - [x] refactor-model-create-base-class – _Create an abstract BaseModel class with a defined interface (save, load, forward, device, etc.)._
  - [x] refactor-eegmodel-to-use-basemodel – _Refactor EEGInceptionERPModel to inherit from BaseModel and implement its interface._
  - [x] refactor-train-model-script – _Update train_model.py to use the refactored EEGInceptionERPModel (inheriting from BaseModel)._
  - [ ] refactor-model-documentation – _Ensure all model classes have NASA-grade docstrings and input validation._
  - [ ] refactor-model-file-structure – _Ensure each model class resides in its own file._

## Review / Testing

## Done ✅
