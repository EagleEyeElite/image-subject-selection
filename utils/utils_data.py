import random
import numpy as np
import pandas as pd
import os
from datetime import datetime

def get_random_samples(dataset, sample_size):
    """
    Select a random sample of images from the dataset.

    Args:
    - dataset: The dataset from which to sample.
    - sample_size: The number of samples to select.

    Returns:
    - sample_indices: List of indices for the selected samples.
    """
    sample_indices = random.sample(range(len(dataset)), sample_size)
    return sample_indices


def get_random_samples_filtered(indices, sample_size):
    return random.sample(indices, sample_size)

class DatasetToDataFrameConverter:
    def __init__(self, dataset, labels_dict):
        """
        Initialize the DatasetToDataFrameConverter.

        Args:
            dataset (Dataset): The dataset object (e.g., pascal_ds) containing images and masks.
            labels_dict (dict): A dictionary mapping label indices to label names.
        """
        self.dataset = dataset
        self.labels_dict = labels_dict

    def convert(self):
        """
        Process the dataset to extract the mask information and create a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing dataset indices, labels, label count, and label names.
        """
        data = []

        for idx, (_, mask) in enumerate(self.dataset):
            mask = np.array(mask)
            labels = np.unique(mask)
            labels = labels[(labels != 0) & (labels != 255)]  # Filter out background and undefined

            data.append({
                "dataset_idx": idx,
                "labels": labels,
                "label_count": len(labels),
                "label_names": [self.labels_dict[label] for label in labels],
            })

        # Create and return a pandas DataFrame
        return pd.DataFrame(data)

def filter_dataset_by_classes(dataset, exclude_classes, pascal_labels):
    filtered_indices = []
    
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask = np.array(mask)
        
        # Extract unique labels, ignoring 0 (background) and 255 (border/undefined)
        labels = np.unique(mask)
        labels = labels[(labels != 0) & (labels != 255)]
        
        # Skip if the image has more than 1 label (to match the DataFrame filter: label_count == 1)
        if len(labels) != 1:
            continue
        
        # Convert the labels to label names using pascal_labels mapping
        label_names = [pascal_labels[label] for label in labels]
        
        # Check if any excluded class is present in the label_names
        if not any(cls in label_names for cls in exclude_classes):
            filtered_indices.append(idx)
    
    return filtered_indices



def extract_true_label(target_mask, label_mapping):
    """
    Extract the true label from the segmentation mask using a provided label mapping.
    
    Args:
        target_mask (np.ndarray or tensor): The target segmentation mask.
        label_mapping (dict): A dictionary mapping label indices to human-readable class names.
        
    Returns:
        str: The human-readable label(s) extracted from the mask.
    """
    target_mask = np.array(target_mask)  # Convert to numpy array if it's a tensor
    unique_labels = np.unique(target_mask)
    unique_labels = unique_labels[(unique_labels != 0) & (unique_labels != 255)]  # Exclude background (0) and ignored regions (255)

    # Map the unique labels to human-readable class names
    label_names = [label_mapping.get(label, "Unknown") for label in unique_labels]

    return ", ".join(label_names) if label_names else "Unknown"


def save_dataframe_to_csv(df, file_name, folder_path):
    """
    Save the given DataFrame as a CSV file with the current date and time in the filename.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        folder_path (str): The folder where the CSV file should be saved.
    
    Raises:
        FileNotFoundError: If the specified folder does not exist.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist. Please create it first.")
    
    # Get the current date and time in the format: "YYYY-MM-DD_HH-MM-SS"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Construct the full file path with the date, time, and filename
    file_path = os.path.join(folder_path, f"{file_name}_{current_time}.csv")
    
    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)
    
    print(f"DataFrame saved as CSV at: {file_path}")
