#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A streamlined script for evaluating a pre-trained ECG classification model.

This script loads a model and a test dataset, then computes evaluation metrics
at the individual ECG segment level.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- Model Imports ---
# Directly import the final, self-contained model class
from model.WeightModel import Model

# --- Utility Imports ---
# Assuming this function is in your utils, as it's a required preprocessing step
from utils.Augmentation import compress_qrs_peaks

# --- Helper Functions ---

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_ecg_data_from_npz(file_path):
    """
    Loads ECG data and labels from a single .npz file.
    
    Args:
        file_path (str): The path to the .npz file.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing data and labels as tensors.
    """
    print(f"Loading data from {file_path}...")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    
    with np.load(file_path) as npz_file:
        if 'data' not in npz_file or 'label' not in npz_file:
            raise ValueError(f"File {file_path} must contain 'data' and 'label' keys.")
        
        data = npz_file['data']
        if data.ndim == 3:
            data = data.squeeze(2)
        labels = npz_file['label']

    # Apply any necessary preprocessing like QRS compression
    # This function is assumed to be available from your utils
    data = compress_qrs_peaks(data, threshold=0.5, compression_ratio=0.3)
    
    # Convert to PyTorch tensors and ensure correct shape/type
    # Shape for data: [num_samples, sequence_length, num_channels=1]
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    print(f"Loaded {data_tensor.shape[0]} samples.")
    return data_tensor, labels_tensor

def print_metrics(metrics, dataset_name="Test Set"):
    """Prints evaluation metrics in a formatted way."""
    print(f"\n--- {dataset_name} Evaluation Results ---")
    print(f"- ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"- Accuracy:  {metrics['accuracy']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall:    {metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("------------------------------------")

# --- Core Evaluation Function ---

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluates the model on a dataset at the individual segment level.
    """
    model.eval()
    all_true_labels = []
    all_pred_scores = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for data, labels in progress:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            probabilities = torch.sigmoid(outputs.squeeze(-1))

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_scores.extend(probabilities.cpu().numpy())

    # Convert probability scores to binary predictions
    all_pred_labels = [1 if score >= threshold else 0 for score in all_pred_scores]

    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(all_true_labels, all_pred_scores),
        'accuracy': accuracy_score(all_true_labels, all_pred_labels),
        'precision': precision_score(all_true_labels, all_pred_labels, zero_division=0),
        'recall': recall_score(all_true_labels, all_pred_labels, zero_division=0),
        'confusion_matrix': confusion_matrix(all_true_labels, all_pred_labels),
        'classification_report': classification_report(
            all_true_labels, all_pred_labels,
            target_names=["Class 0", "Class 1"],
            zero_division=0
        )
    }
    return metrics

def run_evaluation(args, device):
    """
    Main function to set up model, data, and run the evaluation pipeline.
    """
    # 1. Load Data
    test_data, test_labels = load_ecg_data_from_npz(args.test_data_path)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize Model Architecture using the imported 'Model' class
    # The 'Model' class handles its own sub-module creation and setup.
    print("Initializing model...")
    model = Model(device, load_path=args.unet_pretrain_path).to(device)
    
    # 3. Load Model Weights
    print(f"Loading final model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 4. Run Evaluation
    test_metrics = evaluate_model(model, test_loader, device, threshold=0.5)
    print_metrics(test_metrics, "Final Test Set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Classification Model Evaluation Script.")
    
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to the test dataset file (.npz).')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved final model weights file (.pth).')
    parser.add_argument('--unet_pretrain_path', type=str, default=None,
                        help='(Optional) Path to pre-trained UNet weights needed for model initialization.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for evaluation.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Run the main evaluation process
    run_evaluation(args, device)

