import torch
import numpy as np

class CrossCorrelLoss(torch.nn.Module):
    def __init__(self, true_data, name="CrossCorrelationLoss"):
        # Call the constructor of the parent class
        super(CrossCorrelLoss, self).__init__()
        # Store the true data
        self.true_data = true_data
        # Store the name of the loss function
        self.name = name
        # Initialize the success flag
        self.success = torch.tensor(False)

    def forward(self, gen_data):
        # Calculate the covariance matrix of the true data
        cov_true = self.calculate_covariance(self.true_data)
        # Calculate the covariance matrix of the generated data
        cov_gen = self.calculate_covariance(gen_data)

        # Calculate the correlation measure
        correlation = self.calculate_correlation_measure(cov_true, cov_gen)

        # Set a threshold for success determination
        threshold = 0.1  # Adjustable threshold
        if correlation < threshold:
            self.success = torch.tensor(True)

        return correlation

    def calculate_covariance(self, X):
        """
        Calculate the covariance matrix
        :param X: Shape is (batch, seq_len, feature)
        :return: Covariance matrix, shape is (batch, feature, feature)
        """
        batch, seq_len, feature = X.shape

        # Calculate the covariance using matrix operations
        X_centered = X - X.mean(dim=1, keepdim=True)  # Remove the mean
        cov_matrix = torch.bmm(X_centered.transpose(1, 2), X_centered) / seq_len

        return cov_matrix

    def calculate_correlation_measure(self, cov_r, cov_f):
        """
        Calculate the correlation measure
        :param cov_r: Covariance matrix of the real data
        :param cov_f: Covariance matrix of the generated data
        :return: Correlation measure
        """
        # Calculate the standardized covariance
        cov_r_ij = cov_r / torch.sqrt(cov_r.diagonal(dim1=1, dim2=2))  # The diagonal is the covariance itself
        cov_f_ij = cov_f / torch.sqrt(cov_f.diagonal(dim1=1, dim2=2))

        # Calculate the absolute difference
        correlation_diff = torch.abs(cov_r_ij - cov_f_ij)

        # Calculate the overall correlation measure
        correlation_measure = correlation_diff.mean()

        return correlation_measure