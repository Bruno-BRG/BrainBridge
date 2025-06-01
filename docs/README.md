# Project Documentation

This document provides an overview of the project structure and key components.

## Data Loading (`src/data/data_loader.py`)

The `data_loader.py` module is responsible for loading, preprocessing, and windowing the EEG data from the PhysioNet Motor Imagery Dataset.

### PlantUML Diagram

```plantuml
@startuml
class BCIDataLoader {
  + data_path: str
  + subjects: Optional[List[int]]
  + runs: Optional[List[int]]
  + sample_rate: int
  + n_channels: int
  + event_mapping: Dict[str, int]
  + channel_names: List[str]
  + __init__(data_path: str, subjects: Optional[List[int]], runs: Optional[List[int]], sample_rate: int, n_channels: int)
  + get_available_subjects() -> List[int]
  + load_csv_data(subject_id: int, run: int) -> Tuple[np.ndarray, List[Tuple[int, str]]]
  + preprocess_data(eeg_data: np.ndarray, lowcut: float, highcut: float, notch_freq: float, apply_standardization: bool) -> np.ndarray
  + create_windows(eeg_data: np.ndarray, events: List[Tuple[int, str]], window_length: float, baseline_length: float, overlap: float) -> Tuple[np.ndarray, np.ndarray]
  + load_subject_data(subject_id: int, preprocess: bool, create_windows_flag: bool, **kwargs) -> Tuple[np.ndarray, np.ndarray]
  + load_all_subjects(**kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
}

class BCIDataset {
  + windows: torch.Tensor
  + labels: torch.Tensor
  + transform: Optional[callable]
  + augment: bool
  + __init__(windows: np.ndarray, labels: np.ndarray, transform: Optional[callable], augment: bool)
  + __len__() -> int
  + __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]
  + _augment_window(window: torch.Tensor) -> torch.Tensor
  + get_class_weights() -> torch.Tensor
}

BCIDataLoader ..> BCIDataset : creates
DataLoader "1" *-- "1" BCIDataset : uses

package torch.utils.data {
  class Dataset
  class DataLoader
}

BCIDataset --|> Dataset
@enduml
```

### Class Descriptions

#### `BCIDataLoader`
Handles loading CSV files from the PhysioNet Motor Imagery dataset, extracting events, preprocessing (filtering, standardization), and creating windowed data suitable for CNN training.

#### `BCIDataset`
A PyTorch `Dataset` class that wraps the windowed EEG data and labels, making it compatible with PyTorch `DataLoader` for batching and training. It also supports optional data augmentation and transformations.

### Key Functions

-   `create_data_loaders(...)`: A utility function that takes a `BCIDataLoader` instance, loads all subject data, splits it into training, validation, and test sets (subject-wise), and returns PyTorch `DataLoader` instances for each set.

