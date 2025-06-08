# BCI System Architecture Documentation

## Overview

The BCI EEG Motor Imagery Classification System implements a sophisticated architecture designed for both research and clinical applications, with clear separation of concerns and modular design principles.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[EEG Hardware]
        B[PhysioNet Dataset]
        C[Patient Recordings]
    end
    
    subgraph "Data Acquisition"
        D[PyLSL Stream]
        E[CSV Data Loader]
        F[Patient Data Manager]
    end
    
    subgraph "Processing Pipeline"
        G[Signal Preprocessing]
        H[Feature Extraction]
        I[Data Augmentation]
    end
    
    subgraph "Machine Learning"
        J[EEGInceptionERP Model]
        K[Training Engine]
        L[Fine-Tuning Engine]
        M[Real-Time Inference]
    end
    
    subgraph "User Interfaces"
        N[DevTools GUI]
        O[EndUser GUI]
        P[Visualization]
    end
    
    subgraph "Output Systems"
        Q[VR Interface]
        R[Model Reports]
        S[Patient Reports]
    end
    
    A --> D
    B --> E
    C --> F
    
    D --> G
    E --> G
    F --> G
    
    G --> H
    H --> I
    I --> J
    
    J --> K
    J --> L
    J --> M
    
    K --> N
    L --> O
    M --> Q
    
    N --> P
    O --> P
    P --> R
    P --> S
```

## Component Architecture

### 1. DevTools Architecture

```mermaid
graph LR
    subgraph "DevTools Interface"
        A[Main GUI] --> B[Data Management Tab]
        A --> C[Training Tab]
        A --> D[Fine-Tuning Tab]
        A --> E[PyLSL Tab]
    end
    
    subgraph "Core Components"
        F[Data Loader]
        G[Model Trainer]
        H[Fine-Tuner]
        I[Stream Reader]
    end
    
    subgraph "Visualization"
        J[Plot Canvas]
        K[Signal Viewer]
        L[Training Plots]
    end
    
    B --> F
    C --> G
    D --> H
    E --> I
    
    F --> J
    G --> L
    H --> L
    I --> K
```

### 2. EndUser Architecture

```mermaid
graph LR
    subgraph "EndUser Interface"
        A[Main GUI] --> B[Patient Management]
        A --> C[PyLSL Recording]
    end
    
    subgraph "Patient System"
        D[Patient Registry]
        E[Patient Data Manager]
        F[Session Manager]
    end
    
    subgraph "Real-Time System"
        G[Stream Processor]
        H[Model Inference]
        I[VR Controller]
    end
    
    B --> D
    B --> E
    C --> F
    C --> G
    
    E --> H
    G --> H
    H --> I
```

## Data Flow Architecture

### Training Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant GUI as DevTools GUI
    participant DL as Data Loader
    participant PP as Preprocessor
    participant T as Trainer
    participant M as Model
    
    U->>GUI: Load Dataset
    GUI->>DL: Load EEG Files
    DL->>PP: Raw EEG Data
    PP->>T: Processed Data
    T->>M: Training
    M->>GUI: Training Results
    GUI->>U: Display Results
```

### Real-Time Processing Flow

```mermaid
sequenceDiagram
    participant H as EEG Hardware
    participant LSL as PyLSL
    participant RP as Real-Time Processor
    participant M as Model
    participant VR as VR Interface
    participant P as Patient
    
    H->>LSL: EEG Stream
    LSL->>RP: Raw Samples
    RP->>RP: Filter & Window
    RP->>M: Processed Data
    M->>VR: Classification
    VR->>P: Feedback
```

## Module Dependencies

### Core Dependencies

```mermaid
graph TD
    A[launch_bci.py] --> B[DevTools.main_gui]
    A --> C[EndUser.main_gui]
    
    B --> D[src.data.data_loader]
    B --> E[src.model.eeg_inception_erp]
    B --> F[src.UI.plot_canvas]
    
    C --> G[EndUser.src.data.patient_data_manager]
    C --> H[src.model.realtime_inference]
    
    D --> I[External: pandas, numpy]
    E --> J[External: torch, braindecode]
    F --> K[External: matplotlib, PyQt5]
    G --> L[External: json, pathlib]
    H --> M[External: pylsl]
```

## Security Architecture

### Data Protection

```mermaid
graph TB
    subgraph "Data Security"
        A[Local Storage Only]
        B[File Permissions]
        C[Patient Anonymization]
        D[Audit Logging]
    end
    
    subgraph "Access Control"
        E[User Authentication]
        F[Role-Based Access]
        G[Session Management]
    end
    
    subgraph "Compliance"
        H[HIPAA Guidelines]
        I[Data Retention Policies]
        J[Patient Consent]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
```

## Performance Architecture

### System Performance Considerations

| Component | Target Performance | Current Implementation |
|-----------|-------------------|----------------------|
| Data Loading | < 5s for 100MB dataset | Optimized pandas/numpy |
| Model Training | < 30 epochs convergence | Early stopping, LR scheduling |
| Real-Time Inference | < 100ms latency | PyTorch inference, GPU acceleration |
| GUI Responsiveness | < 100ms UI updates | Qt threading, async processing |
| Memory Usage | < 4GB RAM | Efficient data structures |

### Scalability Design

```mermaid
graph LR
    subgraph "Horizontal Scaling"
        A[Multiple Patients]
        B[Parallel Training]
        C[Distributed Processing]
    end
    
    subgraph "Vertical Scaling"
        D[GPU Acceleration]
        E[Memory Optimization]
        F[CPU Optimization]
    end
    
    A --> D
    B --> E
    C --> F
```

## Deployment Architecture

### Development Environment

```mermaid
graph TB
    subgraph "Development Setup"
        A[Python 3.8+]
        B[Virtual Environment]
        C[Development Dependencies]
        D[Testing Framework]
    end
    
    B --> A
    C --> B
    D --> C
    
    subgraph "Development Tools"
        E[IDE/Editor]
        F[Version Control]
        G[Debugging Tools]
        H[Profiling Tools]
    end
    
    E --> A
    F --> A
    G --> A
    H --> A
```

### Production Deployment

```mermaid
graph TB
    subgraph "Clinical Environment"
        A[Windows/Linux/Mac]
        B[Python Runtime]
        C[Application Bundle]
        D[Patient Data Storage]
    end
    
    B --> A
    C --> B
    D --> A
    
    subgraph "Configuration"
        E[System Settings]
        F[Model Parameters]
        G[Patient Configuration]
        H[Hardware Setup]
    end
    
    E --> C
    F --> C
    G --> D
    H --> A
```

## Integration Points

### External System Integration

| System | Interface | Protocol | Purpose |
|--------|-----------|----------|---------|
| EEG Hardware | PyLSL | TCP/IP | Real-time data streaming |
| VR Systems | Custom API | TCP/UDP | Control interface |
| Database Systems | File I/O | CSV/JSON | Data persistence |
| Cloud Services | REST API | HTTPS | Data backup (optional) |

### API Endpoints

#### Internal APIs

```python
# Data Management API
class DataManager:
    def load_dataset(path: str) -> Dataset
    def preprocess_data(data: np.ndarray) -> np.ndarray
    def save_processed_data(data: np.ndarray, path: str) -> None

# Model Management API  
class ModelManager:
    def train_model(data: Dataset, config: dict) -> Model
    def fine_tune_model(model: Model, data: Dataset) -> Model
    def evaluate_model(model: Model, data: Dataset) -> dict

# Patient Management API
class PatientManager:
    def register_patient(info: dict) -> str
    def get_patient_data(patient_id: str) -> dict
    def update_patient_data(patient_id: str, data: dict) -> None
```

## Error Handling Architecture

### Error Classification

```mermaid
graph TD
    A[System Errors] --> B[Hardware Errors]
    A --> C[Software Errors]
    A --> D[Data Errors]
    
    B --> E[EEG Connection Lost]
    B --> F[Hardware Malfunction]
    
    C --> G[Model Loading Failed]
    C --> H[GUI Crash]
    C --> I[Memory Error]
    
    D --> J[Corrupted Files]
    D --> K[Invalid Format]
    D --> L[Missing Data]
```

### Recovery Strategies

| Error Type | Recovery Strategy | User Impact |
|------------|------------------|-------------|
| Connection Lost | Auto-reconnect with backoff | Minimal - transparent |
| Model Loading Failed | Fallback to default model | Moderate - notification |
| Data Corruption | Data validation and repair | High - user intervention |
| Memory Error | Garbage collection and optimization | Moderate - performance impact |
| GUI Crash | Graceful restart with state recovery | High - data loss prevention |

## Monitoring and Logging

### System Monitoring

```mermaid
graph LR
    subgraph "Performance Metrics"
        A[CPU Usage]
        B[Memory Usage]
        C[GPU Usage]
        D[Network I/O]
    end
    
    subgraph "Application Metrics"
        E[Model Accuracy]
        F[Inference Latency]
        G[Data Processing Speed]
        H[User Interactions]
    end
    
    subgraph "System Health"
        I[Error Rates]
        J[Uptime]
        K[Resource Utilization]
        L[Performance Trends]
    end
    
    A --> I
    B --> J
    C --> K
    D --> L
    E --> I
    F --> J
    G --> K
    H --> L
```

This architecture documentation provides a comprehensive view of the BCI system's design, ensuring maintainability, scalability, and reliability for both research and clinical applications.
