import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


def train_test_divide(data_x, data_x_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    
    Args:
      - data_x: original data
      - data_x_hat: generated data
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
  
    return train_x, train_x_hat, test_x, test_x_hat


def batch_generator(data, batch_size):
    """Mini-batch generator.

    Args:
        - data: time-series data
        - batch_size: the number of samples in each batch

    Returns:
        - X_mb: time-series data in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]

    return X_mb


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, last_states = self.gru(x)
        last_states = last_states.squeeze(0)
        y_hat_logit = self.fc(last_states)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat



def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Network parameters
    hidden_dim = 16
    iterations = 2000
    batch_size = 64

    # Initialize the discriminator
    discriminator = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(discriminator.parameters())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)

    # Training step
    for itt in tqdm(range(iterations), desc='training', total=iterations):
        # Batch setting
        X_mb = batch_generator(train_x, batch_size)
        X_hat_mb = batch_generator(train_x_hat, batch_size)

        # Convert to tensors
        X_mb = torch.tensor(X_mb, dtype=torch.float32).to(device)
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32).to(device)

        # Train discriminator
        optimizer.zero_grad()

        y_logit_real, y_pred_real = discriminator(X_mb)
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)

        d_loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real).to(device))
        d_loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake).to(device))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optimizer.step()

    # Convert test data to tensors
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32).to(device)

    # Test the performance on the testing set
    with torch.no_grad():
        y_pred_real_curr, _ = discriminator(test_x)
        y_pred_fake_curr, _ = discriminator(test_x_hat)

    y_pred_real_curr = torch.sigmoid(y_pred_real_curr).cpu().numpy()
    y_pred_fake_curr = torch.sigmoid(y_pred_fake_curr).cpu().numpy()

    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    fake_acc = accuracy_score(np.zeros([len(y_pred_fake_curr), ]), (y_pred_fake_curr > 0.5))
    real_acc = accuracy_score(np.ones([len(y_pred_real_curr), ]), (y_pred_real_curr > 0.5))

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc
