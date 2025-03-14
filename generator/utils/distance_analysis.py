import numpy as np
from scipy.spatial.distance import euclidean, cdist
from tqdm import tqdm
import torch
from tqdm.auto import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compute_mmd(x, y, kernel='rbf', gamma=1.0, degree=3, coef0=1.0, device='cpu'):
    """
    Compute the Maximum Mean Discrepancy (MMD) distance between x and y using PyTorch.

    Parameters:
    - x: Tensor, shape [batch_size, feature_dim].
    - y: Tensor, shape [batch_size, feature_dim].
    - kernel: Kernel type, supports 'rbf', 'linear', 'laplacian', 'polynomial', 'sigmoid'.
    - gamma: Parameter for RBF or Laplacian kernel.
    - degree: Degree of the polynomial kernel.
    - coef0: Bias for polynomial and Sigmoid kernels.
    - device: Computing device, 'cpu' or 'cuda'.

    Returns:
    - mmd: Scalar, MMD distance.
    """
    x = x.to(device)
    y = y.to(device)

    if kernel == 'rbf':
        # Compute squared Euclidean distances
        xx = torch.cdist(x, x, p=2) ** 2
        yy = torch.cdist(y, y, p=2) ** 2
        xy = torch.cdist(x, y, p=2) ** 2

        # Compute RBF kernel matrices
        K_xx = torch.exp(-gamma * xx)
        K_yy = torch.exp(-gamma * yy)
        K_xy = torch.exp(-gamma * xy)
    elif kernel == 'linear':
        # Compute linear kernel matrices
        K_xx = torch.matmul(x, x.T)
        K_yy = torch.matmul(y, y.T)
        K_xy = torch.matmul(x, y.T)
    elif kernel == 'laplacian':
        # Compute L1 distances
        xx = torch.cdist(x, x, p=1)
        yy = torch.cdist(y, y, p=1)
        xy = torch.cdist(x, y, p=1)

        # Compute Laplacian kernel matrices
        K_xx = torch.exp(-gamma * xx)
        K_yy = torch.exp(-gamma * yy)
        K_xy = torch.exp(-gamma * xy)
    elif kernel == 'polynomial':
        # Compute polynomial kernel matrices
        K_xx = (torch.matmul(x, x.T) + coef0) ** degree
        K_yy = (torch.matmul(y, y.T) + coef0) ** degree
        K_xy = (torch.matmul(x, y.T) + coef0) ** degree
    elif kernel == 'sigmoid':
        # Compute Sigmoid kernel matrices
        K_xx = torch.tanh(gamma * torch.matmul(x, x.T) + coef0)
        K_yy = torch.tanh(gamma * torch.matmul(y, y.T) + coef0)
        K_xy = torch.tanh(gamma * torch.matmul(x, y.T) + coef0)
    else:
        raise ValueError("Unsupported kernel type. Choose 'rbf', 'linear', 'laplacian', 'polynomial', or 'sigmoid'.")

    # Compute MMD
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def compute_ed(seq1, seq2):
    """
    Compute the Euclidean distance between two sequences.

    Parameters:
    - seq1: NumPy array, shape [seq_len] or [seq_len, feature_dim].
    - seq2: NumPy array, shape [seq_len] or [seq_len, feature_dim].

    Returns:
    - ed: Scalar, Euclidean distance.
    """
    return euclidean(seq1, seq2)


def compute_dtw(seq1, seq2, use_cuda=False):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences using fastdtw.

    Parameters:
    - seq1: NumPy array, shape [seq_len] or [seq_len, feature_dim].
    - seq2: NumPy array, shape [seq_len] or [seq_len, feature_dim].
    - use_cuda: Whether to use CUDA (not implemented as fastdtw does not support GPU acceleration).

    Returns:
    - distance: Scalar, DTW distance.
    """
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance


def compute_ed_batch(real_data, gen_data):
    """
    Compute batch Euclidean distances using NumPy.

    Parameters:
    - real_data: NumPy array, shape [num_samples, feature_dim].
    - gen_data: NumPy array, shape [num_samples, feature_dim].

    Returns:
    - ed_values: NumPy array, shape [num_samples], ED distances for each sample pair.
    """
    return np.linalg.norm(real_data - gen_data, axis=1)


def compute_dtw_batch(real_data, gen_data):
    """
    Compute batch DTW distances.

    Parameters:
    - real_data: NumPy array, shape [num_samples, seq_len] or [num_samples, seq_len, feature_dim].
    - gen_data: NumPy array, shape [num_samples, seq_len] or [num_samples, seq_len, feature_dim].

    Returns:
    - dtw_distances: NumPy array, shape [num_samples], DTW distances for each sample pair.
    """
    dtw_distances = []
    for r, g in tqdm(zip(real_data, gen_data), desc="Calculating DTW distances", total=len(real_data)):
        # Ensure each sequence has shape [seq_len, 1]
        r = r[:, None] if r.ndim == 1 else r
        g = g[:, None] if g.ndim == 1 else g
        try:
            distance = compute_dtw(r, g)
        except Exception as e:
            print(f"Error computing DTW for a pair: {e}")
            distance = np.nan  # Or choose another appropriate default value
        dtw_distances.append(distance)
    return np.array(dtw_distances)


def distance_analysis(real_data, gen_data, kernel='rbf', gamma=0.1, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=64):
    gen_data = gen_data
    gen_data = gen_data[:len(real_data), :]
    # real_data = real_data[:len(gen_data), :]

    assert len(real_data) == len(gen_data), "The lengths of real_data and gen_data must be the same."
    print(f"Real data shape: {real_data.shape}, Generated data shape: {gen_data.shape}")
    print(f"Using device: {device}")

    # Convert data to PyTorch tensors and move to the device
    real_tensor = torch.tensor(real_data, dtype=torch.float32).to(device)
    gen_tensor = torch.tensor(gen_data, dtype=torch.float32).to(device)

    # Compute MMD
    print("Calculating MMD distances...")
    num_samples = real_tensor.shape[0]
    mmd_values = []
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Calculating MMD distances"):
            x_batch = real_tensor[i:i + batch_size]
            y_batch = gen_tensor[i:i + batch_size]
            mmd_batch = compute_mmd(x_batch, y_batch, kernel=kernel, gamma=gamma, device=device)
            mmd_values.append(mmd_batch.cpu().numpy())
    mmd_value = np.mean(mmd_values)
    print(f"Average MMD between real and generated data: {mmd_value:.8f}")

    # Compute ED
    print("Calculating ED distances...")
    ed_values = compute_ed_batch(real_data, gen_data)
    ed_value = np.mean(ed_values)
    print(f"Average ED between real and generated data: {ed_value:.8f}")

    # Compute DTW
    print("Calculating DTW distances...")
    dtw_distances = compute_dtw_batch(real_data, gen_data)
    dtw_value = np.nanmean(dtw_distances)  # Use nanmean to ignore possible NaNs
    print(f"Average DTW between real and generated data: {dtw_value:.8f}")



def main():
    batch = 64
    seq_len = 2048
    label = 1
    real_data = np.random.rand(batch, seq_len,label)
    gen_data = np.random.rand(batch, seq_len,label)

     
    distance_analysis(real_data, gen_data)


if __name__ == "__main__":
    main()
