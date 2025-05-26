import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from src.data.data_loader import BCIDataLoader

class DataLoadThread(QThread):
    """Thread for loading EEG data asynchronously"""
    progress_updated = pyqtSignal(int)
    data_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, data_path, subjects, runs):
        super().__init__()
        self.data_path = data_path
        self.subjects = subjects
        self.runs = runs

    def run(self):
        try:
            loader = BCIDataLoader(self.data_path, subjects=self.subjects, runs=self.runs)
            total_subjects = len(self.subjects)
            data_dict = {
                'windows': [],
                'labels': [],
                'subject_ids': []
            }

            for i, subject_id in enumerate(self.subjects):
                windows, labels = loader.load_subject_data(subject_id)
                if len(windows) > 0:
                    data_dict['windows'].extend(windows)
                    data_dict['labels'].extend(labels)
                    data_dict['subject_ids'].extend([subject_id] * len(windows))
                progress = int((i + 1) / total_subjects * 100)
                self.progress_updated.emit(progress)

            # Convert lists to numpy arrays before emitting
            if len(data_dict['windows']) > 0:
                data_dict['windows'] = np.array(data_dict['windows'])
                data_dict['labels'] = np.array(data_dict['labels'])
                data_dict['subject_ids'] = np.array(data_dict['subject_ids'])

            self.data_loaded.emit(data_dict)
        except Exception as e:
            self.error_occurred.emit(str(e))
