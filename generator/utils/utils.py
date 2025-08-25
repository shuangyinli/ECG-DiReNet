import torch 
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pywt
import time
import os
from scipy.ndimage import label

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, input, target):
        loss = torch.mean((input - target) ** 2).sum(dim=0)
        print("loss", loss)
        return loss


class EMA():
    def __init__(self, betas):
        super(EMA, self).__init__()
        self.betas = betas

    def update_model_param(self, old_model, current_model):
        for old_param, cur_param in zip(old_model.parameters(), current_model.parameters()):
            old_data, cur_data = old_param.data, cur_param.data
            old_param.data = cur_data if old_data is None else old_data * self.betas + (1 - self.betas) * cur_data



def wavelet_transform(ecg_data):
    coeffs = pywt.wavedec(data=ecg_data, wavelet='sym8', level=8)
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    cD1.fill(0)
    cD2.fill(0)
    coeffs = [cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
    for i in range(0, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='sym8')
    return rdata

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def Normalization2(x):
    mean_val = np.mean(x)
    max_val = np.max(x)
    min_val = np.min(x)
    return [(float(i) - mean_val) / (max_val - min_val) for i in x]


def smooth_continuous_outliers(data, num_outliers=20, window_size=41):
    smoothed_data = np.copy(data)
    half_window = window_size // 2

    for i in range(num_outliers):
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(data))
        average_value = np.mean(data[start:end])
        smoothed_data[i] = average_value
    return smoothed_data


def save_checkpoint(path, epoch,net,  optimizer, dev_best_loss,iter_step,verbose=True):
    data = {
        'epoch': epoch,
        'iter': iter_step,
        'time': time.time(),
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dev_best_loss':dev_best_loss,
    }

    temp_file = None
    if os.path.exists(path):
        temp_file = path + '.old'
        os.rename(path, temp_file)

    with open(path, 'wb') as fp:
        torch.save(data, fp)
        fp.flush()
        os.fsync(fp.fileno())

    if temp_file is not None:
        os.unlink(path + '.old')




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