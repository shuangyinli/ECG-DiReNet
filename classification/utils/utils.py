import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os
from sklearn.metrics import classification_report

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn_val(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    csv_names = [item[2] for item in batch]
    return data, labels, csv_names


def plot_loss_curves(train_losses, val_losses, lrs):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation Loss (lr={lrs[0]:.1e})')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

def compress_qrs_peaks(signal, threshold=0.4, compression_ratio=0.3):
    """
    Compress the amplitude of QRS peaks in a batch of signals to gradually reduce their magnitude.
    
    Args:
        signal (torch.Tensor): Input signal with shape [batch_size, L].
        threshold (float): Threshold value. Values above/below this will be compressed.
        compression_ratio (float): Compression ratio. Smaller values mean stronger compression.
    
    Returns:
        torch.Tensor: Compressed signal with the same shape as the input [batch_size, L].
    """
    if not isinstance(signal, np.ndarray):
        signal_np = signal.numpy()  # Convert to numpy array with shape [batch_size, L]
    else:
        signal_np = signal
    compressed_signal = signal_np.copy()

    # Iterate over each sample in the batch
    for batch_idx in range(signal_np.shape[0]):  # Iterate over batch_size
        signal_batch = signal_np[batch_idx]
        
        # Identify peak regions
        peaks = np.abs(signal_batch) > threshold

        # Find continuous peak regions using scipy.ndimage.label
        labeled_peaks, num_features = label(peaks)

        for region in range(1, num_features + 1):
            # Get indices of the current peak region
            region_indices = np.where(labeled_peaks == region)[0]

            # Apply compression to this region
            for idx in region_indices:
                if signal_batch[idx] > threshold:  # Compress positive peaks
                    compressed_signal[batch_idx, idx] = threshold + compression_ratio * (signal_batch[idx] - threshold)
                elif signal_batch[idx] < -threshold:  # Compress negative peaks
                    compressed_signal[batch_idx, idx] = -threshold + compression_ratio * (signal_batch[idx] + threshold)

    return torch.tensor(compressed_signal, dtype=torch.float32)
def load_and_combine_npz(file_paths, file_specific_limit=None, resvere=False, shuffle=True, include_csv_names=False):
    data_list = []
    labels_list = []
    csv_names_list = []

    for idx, fp in enumerate(file_paths):
        if not os.path.isfile(fp):
            print(f"Warning: File {fp} does not exist, skipping.")
            continue
        try:
            with np.load(fp) as data:
                if 'data' not in data or 'label' not in data:
                    raise ValueError(f"The file {fp} does not contain 'data' and 'label' keys.")
                data_samples = data['data']
                if data_samples.ndim == 3:
                    data_samples = data_samples.squeeze(2)
                label_samples = data['label']
                original_length = data_samples.shape[0]
                if include_csv_names:
                    csv_names_list.append(data['csv_names'])
                # If there is a specific limit for this file, apply it
                if file_specific_limit and fp in file_specific_limit:
                    if resvere:
                        limit = file_specific_limit[fp]
                        data_samples = data_samples[limit:]
                        label_samples = label_samples[limit:]
                        print(f"  Applied limit to file {fp}: {original_length - limit}/{original_length} data points")
                    else:
                        limit = file_specific_limit[fp]
                        data_samples = data_samples[:limit]
                        label_samples = label_samples[:limit]
                        print(f"  Applied limit to file {fp}: {limit}/{original_length} data points")
                else:
                    print(f"  Loaded {original_length} data points")

                data_list.append(data_samples)
                labels_list.append(label_samples)
        except Exception as e:
            print(f"  Error loading file {fp}: {e}")

    if not data_list:
        raise ValueError("No valid data was loaded. Please check file paths and file contents.")

    combined_data = np.concatenate(data_list, axis=0)
    combined_data = compress_qrs_peaks(combined_data, threshold=0.5, compression_ratio=0.3)

    combined_labels = np.concatenate(labels_list, axis=0)

    combined_csv_names = np.concatenate(csv_names_list, axis=0) if include_csv_names else None

    # Shuffle the data
    if shuffle:
        indices = np.random.permutation(len(combined_data))
        combined_data = combined_data[indices]
        combined_labels = combined_labels[indices]
        if combined_csv_names is not None:
            combined_csv_names = combined_csv_names[indices]

    print(f"All files loaded and combined. Total data points: {combined_data.shape[0]}")
    if include_csv_names:
        return (
            torch.tensor(combined_data, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(combined_labels, dtype=torch.float32),
            combined_csv_names
        )
    else:
        return (
            torch.tensor(combined_data, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(combined_labels, dtype=torch.float32),
            None
        )
    
def compress_qrs_peaks(signal, threshold=0.4, compression_ratio=0.3):
    if not isinstance(signal, np.ndarray):
        signal_np = signal.numpy()   
    else:
        signal_np = signal
    compressed_signal = signal_np.copy()
    for batch_idx in range(signal_np.shape[0]):  
        signal_batch = signal_np[batch_idx]
        
        peaks = np.abs(signal_batch) > threshold

        labeled_peaks, num_features = label(peaks)

        for region in range(1, num_features + 1):
            region_indices = np.where(labeled_peaks == region)[0]

            for idx in region_indices:
                if signal_batch[idx] > threshold:  # Compress positive peaks
                    compressed_signal[batch_idx, idx] = threshold + compression_ratio * (signal_batch[idx] - threshold)
                elif signal_batch[idx] < -threshold:  # Compress negative peaks
                    compressed_signal[batch_idx, idx] = -threshold + compression_ratio * (signal_batch[idx] + threshold)

    return torch.tensor(compressed_signal, dtype=torch.float32)



def compute_pos_weight(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label.item())
    labels = np.array(labels)
    count_pos = np.sum(labels == 1)
    count_neg = np.sum(labels == 0)
    if count_pos == 0:
        pos_weight = torch.tensor(1.0)
    else:
        pos_weight = torch.tensor(count_neg / count_pos)
    return pos_weight

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_combine_npz(file_paths, file_specific_limit=None, resvere=False, shuffle=True, include_csv_names=False):
    data_list = []
    labels_list = []
    csv_names_list = []

    for idx, fp in enumerate(file_paths):
        if not os.path.isfile(fp):
            print(f"Warning: File {fp} does not exist, skipping.")
            continue
        try:
            with np.load(fp) as data:
                if 'data' not in data or 'label' not in data:
                    raise ValueError(f"The file {fp} does not contain 'data' and 'label' keys.")
                data_samples = data['data']
                if data_samples.ndim == 3:
                    data_samples = data_samples.squeeze(2)
                label_samples = data['label']
                original_length = data_samples.shape[0]
                if include_csv_names:
                    csv_names_list.append(data['csv_names'])
                if file_specific_limit and fp in file_specific_limit:
                    if resvere:
                        limit = file_specific_limit[fp]
                        data_samples = data_samples[limit:]
                        label_samples = label_samples[limit:]
                        print(f"  Applied limit to file {fp}: {original_length - limit}/{original_length} data entries")
                    else:
                        limit = file_specific_limit[fp]
                        data_samples = data_samples[:limit]
                        label_samples = label_samples[:limit]
                        print(f"  Applied limit to file {fp}: {limit}/{original_length} data entries")
                else:
                    print(f"  Loaded {original_length} data entries")

                data_list.append(data_samples)
                labels_list.append(label_samples)
        except Exception as e:
            print(f"  Error loading file {fp}: {e}")

    if not data_list:
        raise ValueError("No valid data was loaded. Please check file paths and file contents.")

    combined_data = np.concatenate(data_list, axis=0)
    combined_data = compress_qrs_peaks(combined_data, threshold=0.5, compression_ratio=0.3)

    combined_labels = np.concatenate(labels_list, axis=0)
    combined_csv_names = np.concatenate(csv_names_list, axis=0) if include_csv_names else None

    print(f"All files loaded and combined. Total data entries: {combined_data.shape[0]}")
    if include_csv_names:
        return (
            torch.tensor(combined_data, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(combined_labels, dtype=torch.float32),
            combined_csv_names
        )
    else:
        return (
            torch.tensor(combined_data, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(combined_labels, dtype=torch.float32),
            None
        )
def plot_loss_curves(train_losses, val_losses, lrs):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation Loss (lr={lrs[0]:.1e})')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()


def print_metrics(metrics, dataset_name, verbose=False, patient_true=None, patient_pred=None):
    print(f"\n{dataset_name} Evaluation Results:")
    print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- F1: {metrics['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall: {metrics['recall']:.4f}")

    if verbose:
        print("\nDetailed Classification Report:")
        if patient_true is not None and patient_pred is not None:
            print(classification_report(
                patient_true, patient_pred,
                target_names=["Class 0", "Class 1"],
                zero_division=0
            ))

        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
 
 