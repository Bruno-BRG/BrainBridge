# 🏥 Clinical User Guide

> **Complete guide for medical professionals using the BCI system for patient rehabilitation and EEG analysis**

## 🎯 Overview

The EndUser interface is specifically designed for clinical environments, providing a streamlined, intuitive workflow for medical professionals working with stroke patients, brain injury rehabilitation, and motor imagery therapy.

## 🚀 Getting Started

### System Launch

```bash
# Method 1: Using the batch file (Recommended)
launch_enduser.bat

# Method 2: Using Python directly
python launch_bci.py --mode gui
```

**Upon launch, you'll see:**
- Clean, medical-grade interface
- Patient Management tab (primary)
- OpenBCI Live tab for EEG recording
- Status indicators for system health

## 👥 Patient Management Workflow

### 1. Registering a New Patient

#### Step-by-Step Process:

1. **Navigate to Patient Management Tab**
   - This is the default tab upon system launch
   - Clean interface showing registered patients list

2. **Click "New Patient" Button**
   - Opens patient registration form
   - All required fields clearly marked

3. **Fill Patient Information**

   **Basic Information** (Required):
   ```
   Patient ID:     PAT001, JOHN_DOE_001, etc.
   Full Name:      Complete patient name
   Age:           Patient age in years
   Gender:        Male/Female/Other/Prefer not to say
   ```

   **Clinical Information** (Required):
   ```
   Affected Hand:     Left Hand / Right Hand
   Time Since Onset:  "6 months", "2 years", etc.
   ```

4. **Register Patient**
   - Click "Register Patient" button
   - System creates patient folder structure automatically
   - Confirmation message displayed

#### ✅ What Happens Automatically:
- Patient folder created in `patient_data/{patient_id}/`
- Subfolders organized: `eeg_recordings/`, `models/`, `reports/`, `sessions/`
- Patient added to master registry
- Registration date/time logged

### 2. Selecting an Existing Patient

#### Quick Selection Process:

1. **View Patient List**
   - All registered patients displayed in left panel
   - Format: "PAT001 - John Doe"

2. **Click on Patient**
   - Patient details appear in right panel
   - View clinical information and history

3. **Select for Session**
   - Click "Select Patient" button
   - System automatically configures for selected patient
   - Window title updates to show active patient

#### ✅ Auto-Configuration Features:
- Recording folders automatically set
- Patient-specific model paths configured
- Session tracking initialized
- Other tabs updated with patient context

## 📡 EEG Recording Workflow

### 1. OpenBCI Setup and Connection

#### Prerequisites:
- OpenBCI board properly connected
- LSL streaming software running
- EEG electrodes positioned correctly

#### Connection Process:

1. **Navigate to "OpenBCI Live" Tab**
   - Located next to Patient Management tab
   - Shows current patient status if selected

2. **Verify Patient Selection**
   ```
   Patient Status: Active: PAT001 - John Doe (Auto-configured)
   Recording Folder: Automatically set to patient's EEG recordings folder
   ```

3. **Refresh Available Streams**
   - Click "Refresh Streams" button
   - Available LSL streams will be listed
   - Look for OpenBCI stream (typically shows as "obci_eeg1" or similar)

4. **Start Stream Connection**
   - Select your OpenBCI stream from the list
   - Click "Start" to begin data acquisition
   - Real-time EEG visualization begins

### 2. Recording Motor Imagery Sessions

#### Session Recording Steps:

1. **Prepare Patient**
   - Ensure patient is comfortable and relaxed
   - Explain the motor imagery task
   - Position patient appropriately

2. **Start Recording Session**
   - Click "Start Recording" in the interface
   - CSV file automatically created with timestamp
   - File saved to patient's `eeg_recordings/` folder

3. **Conduct Motor Imagery Tasks**

   **Left Hand Imagery:**
   - Instruct patient to imagine left hand movement
   - Click "Record: Left (T1) - 600 samples" button
   - Green indicator shows active recording
   - 600 samples collected (adjustable)

   **Right Hand Imagery:**
   - Instruct patient to imagine right hand movement
   - Click "Record: Right (T2) - 600 samples" button
   - Blue indicator shows active recording
   - 600 samples collected (adjustable)

4. **Monitor Real-Time Feedback**
   - Live EEG waveforms displayed
   - Signal quality indicators
   - Artifact detection alerts
   - Sample count tracking

5. **Complete Session**
   - Click "Stop Recording" when session complete
   - File automatically saved with session metadata
   - Session added to patient's history

#### 📊 Session Data Organization:
```
patient_data/PAT001/eeg_recordings/
├── 2025-06-08_14-30-15_session.csv
├── 2025-06-08_15-45-22_session.csv
└── session_metadata.json
```

## 📈 Monitoring and Progress Tracking

### Real-Time Monitoring

**During Recording:**
- ✅ **Signal Quality**: Real-time assessment of EEG signal quality
- 📊 **Live Plots**: Multi-channel EEG visualization
- ⚠️ **Artifact Detection**: Automatic detection of movement artifacts
- 📝 **Annotation Status**: Current task being recorded (Left/Right/None)

**Session Statistics:**
- Total samples recorded
- Session duration
- Task distribution (Left vs. Right hand trials)
- Signal quality metrics

### Patient Progress Overview

**Historical Data** (Future Implementation):
- Session completion rates
- Signal quality improvement over time
- Motor imagery classification accuracy trends
- Rehabilitation progress metrics

## 🔧 System Configuration

### Recording Parameters

**Default Settings** (Optimized for Clinical Use):
```
Sampling Rate:     250 Hz (standard for motor imagery)
Recording Length:  600 samples per trial (~2.4 seconds at 250Hz)
File Format:       CSV (OpenBCI compatible)
Channels:          8-64 (depends on OpenBCI setup)
```

**Customizable Parameters:**
- Sample count per trial
- Recording session duration
- File naming conventions
- Auto-save intervals

### Data Management Settings

**Automatic Organization:**
- Patient data isolated by ID
- Chronological session organization
- Automated backup suggestions
- Storage space monitoring

## 🔒 Clinical Compliance

### Data Privacy and Security

**HIPAA Compliance Features:**
- ✅ **Local Storage Only**: No data transmitted to cloud
- ✅ **Patient ID Anonymization**: Configurable ID schemes
- ✅ **Access Control**: User-based access management
- ✅ **Audit Trails**: Complete operation logging
- ✅ **Data Retention**: Configurable retention policies

**Best Practices:**
1. Use anonymized patient IDs when possible
2. Regular data backups to secure storage
3. Restrict system access to authorized personnel
4. Regular security updates and maintenance

### Clinical Workflow Integration

**Electronic Health Records (EHR):**
- Export capabilities for EHR integration
- Standardized report formats
- Session summary generation
- Progress tracking metrics

**Research Compliance:**
- IRB-ready data organization
- Consent management support
- De-identification tools
- Research export formats

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. OpenBCI Connection Problems

**Symptoms:** "No streams available" or connection errors
**Solutions:**
```
1. Check OpenBCI hardware connection (USB)
2. Verify LSL software is running
3. Restart OpenBCI GUI software
4. Click "Refresh Streams" in BCI interface
5. Check USB drivers and permissions
```

#### 2. Poor Signal Quality

**Symptoms:** Noisy EEG signals or artifact warnings
**Solutions:**
```
1. Check electrode contact and gel application
2. Reduce movement artifacts (patient positioning)
3. Check for electrical interference (phone, lights)
4. Verify electrode impedances in OpenBCI GUI
5. Re-seat electrodes if necessary
```

#### 3. Recording Issues

**Symptoms:** Files not saving or incomplete recordings
**Solutions:**
```
1. Verify patient folder permissions
2. Check available disk space
3. Ensure patient is properly selected
4. Restart recording session
5. Check system logs for error messages
```

#### 4. Patient Data Problems

**Symptoms:** Cannot load patient or registration errors
**Solutions:**
```
1. Check patient_data folder permissions
2. Verify patients_registry.json file integrity
3. Ensure proper patient ID format (no special characters)
4. Check for duplicate patient IDs
5. Restart application if needed
```

## 📞 Support and Resources

### Getting Help

**Technical Support:**
- Check system logs in `logs/` directory
- Review error messages for specific guidance
- Contact IT support with log files

**Clinical Support:**
- Consult user manual for detailed procedures
- Review training materials
- Contact clinical application specialist

**Emergency Procedures:**
- System backup and restore procedures
- Data recovery protocols
- Alternative recording methods

### Training Resources

**New User Orientation:**
1. System overview and safety
2. Patient registration procedures
3. EEG recording best practices
4. Data management and compliance
5. Troubleshooting common issues

**Advanced Training:**
1. Signal quality optimization
2. Research data export
3. Custom session protocols
4. System administration

---

## 📋 Quick Reference

### Daily Workflow Checklist

```
□ System startup and health check
□ Patient selection and verification
□ OpenBCI hardware check and connection
□ EEG electrode setup and impedance check
□ Recording session execution
□ Data quality verification
□ Session documentation and notes
□ System shutdown and backup
```

### Emergency Contacts

```
Technical Support:  [Your IT Support Contact]
Clinical Support:   [Your Clinical Contact]
System Admin:       [Your System Administrator]
Emergency:          [Emergency Contact Information]
```

This guide ensures clinical professionals can effectively use the BCI system while maintaining the highest standards of patient care and data security.
