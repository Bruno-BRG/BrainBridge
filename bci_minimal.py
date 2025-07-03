import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EEGNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        # Architecture parameters
        self.F1 = 8  # Number of temporal filters
        self.F2 = 16  # Number of pointwise filters
        self.D = 2   # Depth multiplier
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        # Block 2: Spatial Filter
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (n_channels, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 8), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        self.n_samples = n_samples
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F2 * (n_samples // 32), n_classes)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

class EEGDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

def load_openbci_data(data_folder="eeg_data", n_samples_per_trial=400):
    print("Loading OpenBCI data...")
    data_path = Path(data_folder)
    
    all_trials = []
    all_labels = []
    all_subjects = []
    
    # Event mapping
    event_mapping = {'T1': 0, 'T2': 1}  # T1=left hand, T2=right hand
    
    # Process each subject folder
    subject_folders = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('S')])
    
    for subject_folder in subject_folders:
        subject_id = subject_folder.name
        print(f"Processing {subject_id}...")
        
        # Find CSV files
        csv_files = list(subject_folder.glob("*.csv"))
        if not csv_files:
            continue
            
        for csv_file in csv_files:
            try:
                # Read CSV skipping OpenBCI header
                df = pd.read_csv(csv_file, skiprows=4, low_memory=False)
                
                # Check for required columns
                eeg_columns = [f'EXG Channel {i}' for i in range(16)]
                if not all(col in df.columns for col in eeg_columns):
                    continue
                
                # Extract EEG data
                eeg_data = df[eeg_columns].values.astype(float)
                annotations = df['Annotations'].fillna('').astype(str)
                
                # Find T1/T2 events
                for idx, annotation in enumerate(annotations):
                    if annotation in event_mapping:
                        # Extract trial window
                        start_idx = max(0, idx - 50)  # 50 samples before event
                        end_idx = min(len(eeg_data), start_idx + int(3.2 * 125))  # 3.2s trial
                        
                        if end_idx - start_idx < int(3.2 * 125 * 0.8):  # At least 80% of trial
                            continue
                            
                        # Get trial data and resample
                        trial_data = eeg_data[start_idx:end_idx, :].T
                        if trial_data.shape[1] != n_samples_per_trial:
                            old_time = np.linspace(0, 1, trial_data.shape[1])
                            new_time = np.linspace(0, 1, n_samples_per_trial)
                            trial_resampled = np.zeros((16, n_samples_per_trial))
                            
                            for ch in range(16):
                                f = interp1d(old_time, trial_data[ch, :], kind='linear')
                                trial_resampled[ch, :] = f(new_time)
                            trial_data = trial_resampled
                        
                        all_trials.append(trial_data)
                        all_labels.append(event_mapping[annotation])
                        all_subjects.append(subject_id)
                        
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                continue
    
    if all_trials:
        windows = np.array(all_trials)
        labels = np.array(all_labels)
        subject_ids = np.array(all_subjects)
        
        print(f"Dataset loaded: {len(windows)} trials, {len(np.unique(subject_ids))} subjects")
        return windows, labels, subject_ids
    else:
        raise ValueError("No data loaded!")

def plot_data_summary(windows, labels, subject_ids):
    plt.figure(figsize=(15, 10))
    
    # 1. Example trials
    plt.subplot(2, 2, 1)
    class_0_idx = np.where(labels == 0)[0][0]
    class_1_idx = np.where(labels == 1)[0][0]
    time_axis = np.linspace(0, 3.2, windows.shape[2])
    
    for i in range(4):
        offset = i * 100
        plt.plot(time_axis, windows[class_0_idx, i, :] + offset, 'b-', alpha=0.8, label='T1' if i == 0 else "")
        plt.plot(time_axis, windows[class_1_idx, i, :] + offset, 'r-', alpha=0.8, label='T2' if i == 0 else "")
    
    plt.title('EEG Signals by Class\nBlue: Left hand, Red: Right hand')
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')
    plt.legend()
    
    # 2. Power Spectral Density
    plt.subplot(2, 2, 2)
    fs = 125  # OpenBCI sampling rate
    channel_idx = 7  # Central channel
    
    for class_label in [0, 1]:
        class_trials = windows[labels == class_label]
        freqs, psd = signal.welch(class_trials[:, channel_idx, :].mean(axis=0), fs=fs)
        plt.semilogy(freqs, psd, 'b-' if class_label == 0 else 'r-', 
                    label=f"{'Left' if class_label == 0 else 'Right'} hand")
    
    plt.title(f'Power Spectral Density - Channel {channel_idx}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
    
    # 3. Class distribution
    plt.subplot(2, 2, 3)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.pie(counts, labels=['Left hand', 'Right hand'], autopct='%1.1f%%')
    plt.title('Class Distribution')
    
    # 4. Trials per subject
    plt.subplot(2, 2, 4)
    unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    plt.bar(range(len(unique_subjects)), subject_counts)
    plt.title('Trials per Subject')
    plt.xlabel('Subject')
    plt.ylabel('Number of Trials')
    plt.xticks(range(len(unique_subjects)), unique_subjects, rotation=45)
    
    plt.tight_layout()
    plt.show()

def train_model(windows, labels, subject_ids, n_epochs=50, batch_size=32):
    # Data normalization
    windows = (windows - windows.mean()) / windows.std()
    
    # Train/test split
    unique_subjects = np.unique(subject_ids)
    test_subjects = np.random.choice(unique_subjects, size=max(1, len(unique_subjects)//5), replace=False)
    
    test_mask = np.isin(subject_ids, test_subjects)
    X_train, X_test = windows[~test_mask], windows[test_mask]
    y_train, y_test = labels[~test_mask], labels[test_mask]
    
    print(f"Training set: {len(X_train)} trials")
    print(f"Test set: {len(X_test)} trials")
    
    # Create data loaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = EEGNet(n_channels=16, n_classes=2, n_samples=400).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Evaluation
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total
            print(f'Epoch [{epoch+1}/{n_epochs}] - Test Accuracy: {accuracy:.3f}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Training completed! Best accuracy: {best_accuracy:.3f}")
    return model

def main():
    # Load data
    windows, labels, subject_ids = load_openbci_data()
    
    # Plot data summary
    plot_data_summary(windows, labels, subject_ids)
    
    # Train model
    model = train_model(windows, labels, subject_ids)
    
if __name__ == "__main__":
    main()
