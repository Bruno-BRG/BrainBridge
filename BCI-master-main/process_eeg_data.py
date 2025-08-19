import os
import numpy as np
import pandas as pd

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_csv_file(file_path, output_base_path):
    # Read the CSV file
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path, skiprows=4)  # Skip the header rows
    
    # Get patient ID from the file path
    patient_id = os.path.basename(os.path.dirname(file_path))
    
    # EEG channels columns (0-15)
    eeg_columns = [f'EXG Channel {i}' for i in range(16)]
    
    # Find annotation points
    annotations = df['Annotations'].fillna('')
    
    # Process for each annotation type
    for annotation_type in ['T0', 'T1', 'T2']:
        # Find indices where this annotation appears
        annotation_points = annotations[annotations == annotation_type].index
        
        if len(annotation_points) > 0:
            # Create output directory for this annotation type and patient
            output_dir = os.path.join(output_base_path, annotation_type, patient_id)
            create_directory(output_dir)
            
            # Process each occurrence of this annotation
            for idx, point in enumerate(annotation_points):
                # Get 125 samples after the annotation point
                if point + 125 <= len(df):
                    data_segment = df.loc[point:point+124, eeg_columns].values.T
                    
                    # Save as numpy array with annotation type in filename
                    output_file = os.path.join(output_dir, f'{os.path.basename(file_path)[:-4]}_{annotation_type}_{idx}.npy')
                    np.save(output_file, data_segment)
                    print(f"Saved {output_file}")

def main():
    # Base paths
    eeg_data_path = 'eeg_data'
    output_base_path = 'processed_eeg_data'
    
    # Create main output directory
    create_directory(output_base_path)
    
    # Create directories for each annotation type
    for annotation_type in ['T0', 'T1', 'T2']:
        create_directory(os.path.join(output_base_path, annotation_type))
    
    # Walk through all directories in eeg_data
    for root, dirs, files in os.walk(eeg_data_path):
        for file in files:
            if file.endswith('_csv_openbci.csv'):
                file_path = os.path.join(root, file)
                process_csv_file(file_path, output_base_path)

if __name__ == "__main__":
    main()
    print("Processing completed!")
