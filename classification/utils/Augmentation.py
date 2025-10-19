import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label

def add_gaussian_noise(signal, noise_level=0.02):
    """
    Add Gaussian noise to a signal.

    Args:
        signal (torch.Tensor): Input signal of shape [N, L] or [C, L].
        noise_level (float): Noise scale (std) relative to the signal, default 0.02.

    Returns:
        torch.Tensor: Noisy signal with the same shape as input.
    """
    noise = torch.randn_like(signal) * noise_level
    return signal + noise


def time_shift(signal, shift_max=0.2, num_shifts=3):
    """
    Randomly circularly shift each sequence in a batch.

    For each sample in the batch, generate up to `num_shifts` distinct circular
    shifts drawn uniformly from [-shift_max * L, +shift_max * L].

    Args:
        signal (torch.Tensor): Input batch of shape [batch_size, L].
        shift_max (float): Max shift fraction of sequence length (0 ~ 1), default 0.2.
        num_shifts (int): Number of distinct shifts to produce per sample.

    Returns:
        torch.Tensor: All shifted sequences stacked together with shape
                      [batch_size * num_shifts, L].
    """
    signal = signal.clone()  # Work on a copy to avoid in-place changes
    batch_size, seq_len = signal.size()
    max_shift = max(1, int(shift_max * seq_len))

    shifted_signals = []

    for batch_idx in range(batch_size):
        original_signal = signal[batch_idx].unsqueeze(0)  # Single sample [1, L]
        shifts = set()

        for _ in range(num_shifts):
            # Ensure distinct shift values for this sample
            while True:
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                if shift not in shifts:
                    shifts.add(shift)
                    break

            if shift > 0:
                shifted_signal = torch.cat((original_signal[:, shift:], original_signal[:, :shift]), dim=1)
            elif shift < 0:
                shift = -shift
                shifted_signal = torch.cat((original_signal[:, -shift:], original_signal[:, :-shift]), dim=1)
            else:
                shifted_signal = original_signal

            shifted_signals.append(shifted_signal.squeeze(0))  # Back to [L]

    return torch.stack(shifted_signals, dim=0)


def compress_qrs_peaks(signal, threshold=0.4, compression_ratio=0.3):
    """
    Attenuate high-amplitude QRS peaks by compressing values whose absolute
    magnitude exceeds a threshold.

    Args:
        signal (torch.Tensor or np.ndarray): Input batch of shape [batch_size, L].
        threshold (float): Amplitude threshold; values with |x| > threshold are compressed.
        compression_ratio (float): Compression factor (0 ~ 1). Smaller -> stronger compression.

    Returns:
        torch.Tensor: Peak-compressed signal with shape [batch_size, L].
    """
    if not isinstance(signal, np.ndarray):
        # Convert to numpy array on CPU; expects CPU tensors
        signal_np = signal.numpy()  # [batch_size, L]
    else:
        signal_np = signal

    compressed_signal = signal_np.copy()

    # Process each sample independently
    for batch_idx in range(signal_np.shape[0]):
        signal_batch = signal_np[batch_idx]

        # Locate peak regions where |x| > threshold
        peaks = np.abs(signal_batch) > threshold

        # Connected-component labeling to group contiguous peak regions
        labeled_peaks, num_features = label(peaks)

        for region in range(1, num_features + 1):
            # Indices for this contiguous region
            region_indices = np.where(labeled_peaks == region)[0]

            # Apply piecewise linear compression within the region
            for idx in region_indices:
                if signal_batch[idx] > threshold:  # Compress positive peak
                    compressed_signal[batch_idx, idx] = threshold + compression_ratio * (signal_batch[idx] - threshold)
                elif signal_batch[idx] < -threshold:  # Compress negative peak
                    compressed_signal[batch_idx, idx] = -threshold + compression_ratio * (signal_batch[idx] + threshold)

    # Return as a PyTorch tensor
    return torch.tensor(compressed_signal, dtype=torch.float32)


def frequency_shift(signal, shift_factor=0.5):
    """
    Perform frequency-domain augmentation by slightly perturbing the spectrum.

    Args:
        signal (torch.Tensor): Input batch of shape [batch_size, L].
        shift_factor (float): Standard deviation of Gaussian noise added in the frequency domain.

    Returns:
        torch.Tensor: Augmented signal with shape [batch_size, L].
    """
    # Ensure tensor on CPU and detached; cast to float
    signal = signal.clone().detach().float()
    signal_np = signal.numpy()  # [batch_size, L]
    enhanced_signal = np.zeros_like(signal_np)

    # Process each sample independently
    for batch_idx in range(signal_np.shape[0]):
        signal_batch = signal_np[batch_idx]

        # FFT
        freq_signal = np.fft.fft(signal_batch)

        # Add small Gaussian noise in the frequency domain
        noise = np.random.normal(0, shift_factor, freq_signal.shape)
        freq_signal += noise

        # Inverse FFT and keep the real part
        enhanced_signal[batch_idx] = np.fft.ifft(freq_signal).real

    return torch.tensor(enhanced_signal, dtype=torch.float32)


def augment_signal(signal, noise_level=0.2, shift_max=0.2, threshold=0.45, compression_ratio=0.3):
    """
    Apply a suite of augmentations to a batch of 1D signals.

    This function composes multiple transforms:
      - time shifts (circular)
      - frequency-domain perturbation
      - QRS peak compression
      - combinations of the above

 
    Args:
        signal (torch.Tensor): Input batch of shape [N, L].
        noise_level (float): Noise scale used in frequency-domain perturbation.
        shift_max (float): Max circular shift fraction of sequence length.
        threshold (float): Threshold for QRS peak compression.
        compression_ratio (float): Compression factor for peaks.

    Returns:
        torch.Tensor: A stack of augmented variants; each entry has the same shape
                      as `signal` (i.e., [N, L]). The first entry is the original signal.
    """
    # Start with the original
    augmented_signals = [signal]  # Collector for all augmented variants

    # Step 1: time shifts
    shifted_signal = time_shift(signal, shift_max, num_shifts=3)
    for s in shifted_signal:
        augmented_signals.append(s.unsqueeze(0))  # Ensure shape [1, L] to match typical [N=1, L]

    # Step 2: frequency-domain perturbation on the original
    frequency_shifted_signal = frequency_shift(signal.clone(), shift_factor=noise_level)
    augmented_signals.append(frequency_shifted_signal)

    # Frequency-domain perturbation after time shifts
    frequency_shifted_after_shift = frequency_shift(shifted_signal.clone(), shift_factor=noise_level)
    for s in frequency_shifted_after_shift:
        augmented_signals.append(s.unsqueeze(0))

    # Step 3: QRS peak compression on the original
    compressed_signal = compress_qrs_peaks(signal.clone(), threshold=threshold, compression_ratio=compression_ratio)
    augmented_signals.append(compressed_signal)

    # Compression on the frequency-perturbed variant
    compressed_after_shift = compress_qrs_peaks(frequency_shifted_signal.clone(),
                                                threshold=threshold, compression_ratio=compression_ratio)
    for s in compressed_after_shift:
        augmented_signals.append(s.unsqueeze(0))

    # Compression after (time shift + frequency perturbation)
    compressed_after_shift_and_freq = compress_qrs_peaks(frequency_shifted_after_shift.clone(),
                                                         threshold=threshold, compression_ratio=compression_ratio)
    for s in compressed_after_shift_and_freq:
        augmented_signals.append(s.unsqueeze(0))

    # Sanity check: all variants should have the same shape as the first one
    expected_shape = augmented_signals[0].shape
    for i, s in enumerate(augmented_signals):
        assert s.shape == expected_shape, \
            f"Augmented signal {i} has shape {s.shape}, but expected {expected_shape}"

    return torch.stack(augmented_signals, dim=0)
