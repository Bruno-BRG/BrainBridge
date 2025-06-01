# BCI Fine-Tuning and Real-Time Inference System - Task Manager

## Project Overview

**Objective**: Create a system that fine-tunes pre-trained EEG models with patient-specific data for stroke rehabilitation motor imagery tasks, then deploys them for real-time left/right hand movement detection.

**Target Users**: Stroke patients requiring personalized motor imagery BCI systems.

**Data Pipeline**: PyLSL Stream → Patient Recording → Fine-tuning → Validation → Real-time Inference

---

## Current System Analysis

### ✅ What We Have (Implemented)

#### 1. Core Infrastructure
- **Pre-trained Models**: EEGInceptionERP model with full training pipeline
- **Data Management**: BCIDataLoader for PhysioNet dataset (CSV format)
- **Training System**: K-fold cross-validation with early stopping
- **GUI Framework**: PyQt5 interface with 3 tabs (Data Management, Training, PyLSL)
- **Real-time Components**: PyLSL integration for EEG streaming
- **Model Persistence**: Save/load functionality for trained models

#### 2. Data Processing
- **Windowing**: 400-sample windows for training and inference
- **Preprocessing**: Bandpass filtering (0.5-50Hz), standardization
- **Event Handling**: T0 (rest), T1 (left hand), T2 (right hand) annotations
- **CSV Recording**: Real-time data recording with annotations

#### 3. Model Architecture
- **Base Model**: EEGInceptionERP (Braindecode-based)
- **Input**: (batch_size, n_channels=16, n_times=400)
- **Output**: Binary classification (left/right hand)
- **Features**: Built-in feature extraction capability

#### 4. Real-time Inference
- **RealTimeInferenceProcessor**: Buffer management and sliding window inference
- **Timing**: 400-sample windows every 30ms capability
- **Output**: Prediction with confidence scores

---

## 🚧 Missing Components (To Be Implemented)

### Phase 1: Fine-Tuning Infrastructure (Priority: HIGH)

#### 1.1 Fine-Tuning Module (`src/model/fine_tuning.py`)
```python
class ModelFineTuner:
    - load_pretrained_model()
    - prepare_patient_data() 
    - fine_tune_model()
    - validate_fine_tuned_model()
    - save_fine_tuned_model()
```

**Key Features Needed**:
- Load pre-trained model from `models/` directory
- Freeze early layers, fine-tune final layers
- Patient-specific data preprocessing
- Transfer learning with reduced learning rates
- Validation on held-out patient data
- Model comparison metrics

#### 1.2 Patient Data Management
```python
class PatientDataManager:
    - load_patient_recordings()
    - split_training_validation()
    - prepare_fine_tuning_datasets()
    - calculate_data_quality_metrics()
```

**Requirements**:
- Support for multiple recording sessions per patient
- Automatic train/validation split for patient data
- Data quality assessment (signal quality, artifact detection)
- Session metadata management

#### 1.3 Fine-Tuning GUI Tab (`src/UI/fine_tuning_tab.py`)
```python
class FineTuningTab(QWidget):
    - patient_selection_ui()
    - recording_session_management()
    - fine_tuning_configuration()
    - progress_monitoring()
    - validation_results_display()
```

**UI Components Needed**:
- Patient ID input/selection
- Recording session browser (with preview)
- Training/validation data selection
- Fine-tuning parameters (learning rate, epochs, layers to fine-tune)
- Real-time training progress
- Validation metrics display
- Model comparison tools

### Phase 2: Enhanced Real-Time System (Priority: MEDIUM)

#### 2.1 Real-Time Inference Tab (`src/UI/realtime_tab.py`)
```python
class RealTimeTab(QWidget):
    - model_selection_ui()
    - inference_configuration()
    - real_time_display()
    - performance_monitoring()
```

**Features Needed**:
- Fine-tuned model selection dropdown
- Inference parameters configuration
- Real-time prediction display with confidence
- Performance metrics (accuracy over time, latency)
- Visual feedback for left/right predictions

#### 2.2 Enhanced Real-Time Processor
**Current**: `RealTimeInferenceProcessor` (✅ Exists)
**Enhancements Needed**:
- Model hot-swapping capability
- Adaptive confidence thresholding
- Prediction smoothing/filtering
- Performance metrics collection
- Alert system for low confidence periods

#### 2.3 Model Validation System
```python
class ModelValidator:
    - cross_patient_validation()
    - real_time_accuracy_assessment()
    - confidence_calibration()
    - performance_benchmarking()
```

### Phase 3: Advanced Features (Priority: LOW)

#### 3.1 Reinforcement Learning Foundation (Future)
```python
class RLModelAdapter:
    - online_learning_setup()
    - reward_signal_integration()
    - policy_updates()
    - experience_replay()
```

#### 3.2 Advanced Analytics
- Patient progress tracking
- Model performance analytics
- Session comparison tools
- Recommendation system for training parameters

---

## 📋 Implementation Tasks

### Sprint 1: Fine-Tuning Core (Weeks 1-2)

#### Task 1.1: Create Fine-Tuning Module
- [✅] **File**: `src/model/fine_tuning.py`
- [✅] Implement `ModelFineTuner` class
- [✅] Add `load_pretrained_model()` function
- [ ] Implement transfer learning with layer freezing
- [ ] Add patient-specific data preprocessing
- [ ] Create validation pipeline for fine-tuned models

#### Task 1.2: Patient Data Management
- [ ] **File**: `src/data/patient_data_manager.py`
- [ ] Implement `PatientDataManager` class
- [ ] Add CSV data loading for patient recordings
- [ ] Implement train/validation splitting
- [ ] Add data quality assessment functions

#### Task 1.3: Fine-Tuning GUI Tab
- [ ] **File**: `src/UI/fine_tuning_tab.py`
- [ ] Create basic UI layout with patient selection
- [ ] Add recording session browser
- [ ] Implement fine-tuning parameter controls
- [ ] Add progress monitoring and results display
- [ ] Integrate with main GUI (`main_gui.py`)

### Sprint 2: Real-Time Enhancement (Weeks 3-4)

#### Task 2.1: Real-Time Tab
- [ ] **File**: `src/UI/realtime_tab.py`
- [ ] Create real-time inference interface
- [ ] Add model selection dropdown (fine-tuned models)
- [ ] Implement live prediction display
- [ ] Add confidence threshold controls
- [ ] Create performance monitoring dashboard

#### Task 2.2: Enhanced Inference System
- [ ] **Modify**: `src/model/realtime_inference.py`
- [ ] Add model hot-swapping capability
- [ ] Implement prediction smoothing
- [ ] Add performance metrics collection
- [ ] Create confidence calibration system

#### Task 2.3: Model Validation Framework
- [ ] **File**: `src/model/model_validator.py`
- [ ] Implement validation metrics for fine-tuned models
- [ ] Add real-time accuracy assessment
- [ ] Create model comparison tools
- [ ] Add automated quality checks

### Sprint 3: Integration & Testing (Week 5)

#### Task 3.1: System Integration
- [ ] Integrate all new components into main GUI
- [ ] Update main window with new tabs
- [ ] Test complete workflow: Record → Fine-tune → Validate → Deploy
- [ ] Ensure proper error handling and user feedback

#### Task 3.2: Documentation & Testing
- [ ] Update README with new workflow instructions
- [ ] Create user guide for fine-tuning process
- [ ] Add unit tests for new components
- [ ] Performance testing and optimization

---

## 🔄 Workflow Design

### Fine-Tuning Workflow
1. **Patient Setup**: Create patient ID, configure recording parameters
2. **Data Collection**: Record training sessions via PyLSL (T1/T2 annotations)
3. **Data Preparation**: Load recordings, quality check, train/val split
4. **Model Selection**: Choose pre-trained base model
5. **Fine-Tuning**: Configure parameters, start fine-tuning process
6. **Validation**: Test on held-out patient data, assess performance
7. **Deployment**: If accuracy threshold met, deploy for real-time use

### Real-Time Inference Workflow
1. **Model Loading**: Load fine-tuned patient-specific model
2. **Stream Setup**: Connect to PyLSL EEG stream
3. **Real-Time Processing**: 400-sample windows every 30ms
4. **Prediction**: Output left/right with confidence
5. **Monitoring**: Track accuracy, latency, confidence trends
6. **Adaptation**: Optional online learning adjustments (future)

---

## 📊 Success Metrics

### Fine-Tuning Success Criteria
- [ ] **Accuracy Improvement**: Fine-tuned model shows >10% accuracy improvement over base model on patient data
- [ ] **Training Time**: Fine-tuning completes in <30 minutes for typical dataset
- [ ] **Validation Accuracy**: Achieves >70% accuracy on held-out patient validation data
- [ ] **Robustness**: Consistent performance across multiple recording sessions

### Real-Time Performance Criteria
- [ ] **Latency**: <50ms inference time per 400-sample window
- [ ] **Reliability**: >95% uptime during real-time sessions
- [ ] **Accuracy**: Maintains validation accuracy in real-time deployment
- [ ] **User Experience**: Clear, responsive UI with intuitive controls

---

## 🔧 Technical Specifications

### Data Requirements
- **Window Size**: 400 samples (3.2s at 125Hz)
- **Channels**: 16 EEG channels (standard 10-20 system)
- **Sample Rate**: 125Hz
- **Training Data**: Minimum 100 windows per class per patient
- **Validation Split**: 20% of patient data

### Model Requirements
- **Base Architecture**: EEGInceptionERP (pre-trained on PhysioNet)
- **Fine-Tuning**: Transfer learning with partial layer freezing
- **Output**: Binary classification (left/right hand imagery)
- **Format**: PyTorch `.pth` files with metadata

### Hardware Requirements
- **Memory**: 8GB RAM minimum for training
- **Storage**: 2GB for models and patient data
- **GPU**: Optional but recommended for faster fine-tuning
- **EEG Device**: OpenBCI-compatible with PyLSL support

---

## 🚀 Next Steps

### Immediate Actions (Week 1)
1. **Start with Task 1.1**: Create `fine_tuning.py` module
2. **Set up development branch**: `feature/fine-tuning-system`
3. **Create test data**: Simulate patient recordings for development
4. **Design database schema**: Patient data organization structure

### Dependencies to Resolve
- [ ] Verify pre-trained model compatibility for fine-tuning
- [ ] Test PyLSL recording quality and annotation timing
- [ ] Confirm GUI framework extensibility for new tabs
- [ ] Validate real-time performance with current hardware

---

*Last Updated: June 1, 2025*
*Status: Planning Phase - Ready for Implementation*
