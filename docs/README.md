# BCI EEG Motor Imagery Classification System

> **A Professional Brain-Computer Interface System for Motor Imagery Classification and Rehabilitation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Torch](https://img.shields.io/badge/ML-PyTorch-red.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](#documentation)

## 🎯 Project Overview

This system provides a complete Brain-Computer Interface (BCI) solution for EEG motor imagery classification, designed for both research and clinical rehabilitation applications. It features dual interfaces: a comprehensive **DevTools** environment for researchers and a streamlined **EndUser** interface for clinical professionals.

### Key Features

- **🧠 State-of-the-Art ML**: EEGInceptionERP CNN architecture with fine-tuning capabilities
- **🏥 Clinical-Ready**: Patient management system with HIPAA-compliant data handling
- **📡 Real-Time Processing**: Live EEG streaming via PyLSL with OpenBCI integration
- **🔬 Research Tools**: Comprehensive data analysis, visualization, and model development
- **⚡ High Performance**: Optimized pipeline with CUDA support for GPU acceleration
- **📊 Rich Visualization**: Interactive plots, training curves, and real-time monitoring

## 🏗️ System Architecture

```
🌟 BCI System
├── 🔧 DevTools/          # Research & Development Environment
│   ├── Data Management   # Dataset handling and preprocessing
│   ├── Model Training    # ML model development and training
│   ├── Fine-Tuning       # Patient-specific model adaptation
│   └── Live Analysis     # Real-time EEG stream analysis
│
├── 🏥 EndUser/           # Clinical Interface
│   ├── Patient Mgmt      # Patient registration and data management
│   └── Live Recording    # Real-time EEG acquisition and analysis
│
├── 🧠 Core ML Engine/    # Machine Learning Pipeline
│   ├── EEGInceptionERP   # Deep learning model
│   ├── Signal Processing # EEG preprocessing and filtering
│   └── Real-time Inference # Live prediction engine
│
└── 📊 Data Layer/        # Data Management
    ├── PhysioNet Data    # Public research datasets
    ├── Patient Records   # Clinical data storage
    └── Model Repository  # Trained models and metadata
```

## 🚀 Quick Start

### For Clinical Users (EndUser Interface)

```bash
# Launch clinical interface
python launch_enduser.bat
```

### For Researchers (DevTools Interface)

```bash
# Launch development environment
python launch_bci.py --mode gui
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[🏗️ System Architecture](SYSTEM_ARCHITECTURE.md)** | Technical architecture and design patterns |
| **[⚙️ Installation Guide](INSTALLATION.md)** | Complete setup instructions and dependencies |
| **[👨‍⚕️ Clinical User Guide](USER_GUIDE_CLINICAL.md)** | Step-by-step guide for medical professionals |
| **[🔬 Research User Guide](USER_GUIDE_RESEARCH.md)** | Advanced features for researchers |
| **[🔌 API Documentation](API_DOCUMENTATION.md)** | Complete API reference and examples |
| **[🛠️ Developer Setup](DEVELOPER_SETUP.md)** | Development environment configuration |
| **[🧪 Testing Guide](TESTING.md)** | Comprehensive testing procedures |
| **[📈 Performance Guide](PERFORMANCE.md)** | Optimization and benchmarking |

## 🎯 Use Cases

### Clinical Rehabilitation
- **Stroke Recovery**: Motor imagery training for affected limb rehabilitation
- **Brain Injury**: Neuroplasticity assessment and training protocols
- **Research Studies**: Clinical trial data collection and analysis

### Research Applications
- **Algorithm Development**: New ML models and signal processing techniques
- **Dataset Analysis**: Large-scale EEG data mining and pattern discovery
- **Real-time Systems**: BCI control interface development

## 🔬 Technical Highlights

### Machine Learning
- **EEGInceptionERP Architecture**: State-of-the-art CNN for EEG classification
- **Transfer Learning**: Pre-trained models with patient-specific fine-tuning
- **Real-Time Inference**: Sub-100ms prediction latency for live applications

### Data Processing
- **Signal Preprocessing**: Advanced filtering, artifact removal, and normalization
- **Feature Engineering**: Automated temporal and spectral feature extraction
- **Data Augmentation**: Synthetic data generation for improved generalization

### Clinical Features
- **Patient Management**: Comprehensive clinical data tracking and organization
- **Session Recording**: Structured EEG data collection with metadata
- **Progress Tracking**: Longitudinal analysis and improvement metrics

## 🤝 Contributing

We welcome contributions from researchers, clinicians, and developers! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code standards and best practices
- Testing requirements
- Documentation standards
- Issue reporting and feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **PhysioNet**: For providing the EEG Motor Movement/Imagery Database
- **Braindecode**: For the excellent EEG deep learning framework
- **OpenBCI**: For hardware integration support and community
- **PyTorch Community**: For the robust machine learning foundation

## 📞 Support

- 📧 **Email**: [Your support email]
- 💬 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 **Wiki**: [Project Wiki](https://github.com/your-repo/wiki)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**Built with ❤️ for advancing Brain-Computer Interface technology in research and clinical applications.**
