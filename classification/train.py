import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import random
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse


from collections import defaultdict
import seaborn as sns
from dataset.Dataset import PatientDataset
from utils.utils import compress_qrs_peaks
from utils.loss import FocalLoss,BCEWithLogitsLossLabelSmoothing,CrossEntropyLabelSmoothing,MultiClassFocalLoss
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from utils.utils import set_seed,compress_qrs_peaks,set_seed,compute_pos_weight,load_and_combine_npz,plot_loss_curves,print_metrics
from model.DiffusionUnet import ECGunetChannels 
from model.ECGNet import ECGNet
from model.WeightModel import Model


def collate_fn_val(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    csv_names = [item[2] for item in batch]
    return data, labels, csv_names

def discriminative_score_metrics(train_dataset, val_dataset, test_dataset, device, best_model_path, epochs=60,
                                 batch_size=512, lr=5e-3, weight_decay=1e-4, seed=42, checkpoint_path=None,pretrain_path=None):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn_val  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn_val  
    )

    number_of_diffusions = 1000
    kernel_size = 5
    num_levels = 3
    n_channels = 1
    resolution = 2048

    model = Model(
        number_of_diffusions=number_of_diffusions,
        kernel_size=kernel_size,
        num_levels=num_levels,
        n_channels=n_channels,
        resolution=resolution,
        device=device,
        load_path=pretrain_path
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),   
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.8, 0.999)
    )

    pos_weight = compute_pos_weight(train_dataset)
    pos_weight = pos_weight.to(device)
 
    criterion = FocalLoss(alpha=1.2, gamma=2.0, reduction='mean')

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=8, verbose=True, min_lr=1e-5)

    best_auc = float('-inf')   

    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====\n")

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False)
        for data, labels in train_progress:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(data)   
            loss = criterion(outputs.squeeze(-1), labels)

            preds = (torch.sigmoid(outputs.squeeze(-1)) > 0.5).int()
            correct = (preds == labels.int()).sum().item()
            epoch_correct += correct
            epoch_total += labels.size(0)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            current_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            train_progress.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_accuracy * 100:.2f}%"
            })

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Average Acc: {avg_epoch_acc * 100:.2f}%")
        train_losses.append(avg_epoch_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 1 == 0:
            print(f"\n===== Epoch {epoch + 1}/{epochs} - Validation =====\n")
            val_metrics = evaluate_model(model, val_loader, criterion, device, threshold=0.47)
            print_metrics(val_metrics, "Validation set", verbose=True)

            if val_metrics['roc_auc'] > best_auc:
                best_auc = val_metrics['roc_auc']
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 New best model saved (AUC={best_auc:.4f})")

            scheduler.step(val_metrics['accuracy'])
            val_losses.append(val_metrics['loss'])

    print("\n=== Final Test ===")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, criterion, device, threshold=0.47)
    print_metrics(test_metrics, "Test set", verbose=True)


def evaluate_model(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    all_csv_names = []
    total_loss = 0.0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for data, labels, csv_names in progress:
            data = data.to(device)
            labels = labels.float().to(device)

            outputs = model(data)
            loss = criterion(outputs.squeeze(-1), labels)
            total_loss += loss.item() * data.size(0)

            probabilities = torch.sigmoid(outputs.squeeze(-1))
            preds = (probabilities > threshold).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())
            all_csv_names.extend(csv_names)

    patient_results = defaultdict(lambda: {'scores': [], 'true_label': None})
    for csv_name, true_label, score in zip(all_csv_names, y_true, y_score):
        patient_results[csv_name]['true_label'] = true_label
        patient_results[csv_name]['scores'].append(score)

    patient_true = []
    patient_scores = []
    for csv_name, data in patient_results.items():
        patient_true.append(data['true_label'])
        patient_scores.append(np.mean(data['scores']))

    patient_pred = [1 if score >= threshold else 0 for score in patient_scores]

    metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'roc_auc': roc_auc_score(patient_true, patient_scores),
        'accuracy': accuracy_score(patient_true, patient_pred),
        'confusion_matrix': confusion_matrix(patient_true, patient_pred),
        'precision': precision_score(patient_true, patient_pred),
        'recall': recall_score(patient_true, patient_pred),
        'classification_report': classification_report(
            patient_true, patient_pred,
            target_names=["Class 0", "Class 1"],
            output_dict=True, zero_division=0
        )
    }
    return metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a discriminative model.')

    parser.add_argument('--seed', type=int, default=99, help='Random seed for reproducibility')
    parser.add_argument('--train_paths', nargs='+', default=["../data/train.npz"],
                        help='Paths to training data files')
    parser.add_argument('--generator_train_paths', nargs='+', default=["../data/clean_data.npz","../data/clean_data3.npz","../data/clean_data4.npz"],
                        help='Paths to original training data files')
    parser.add_argument('--npz_result_val', type=str, default="../data/val.npz", help='Path to validation data file')
    parser.add_argument('--npz_result_test', type=str, default="../data/test.npz", help='Path to test data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to load a pre-trained model checkpoint')
    parser.add_argument('--best_model_path', type=str, default='./weight/classification_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to load a pre-trained Unet model ')


    args = parser.parse_args()
    train_file_paths = []
    train_file_paths.extend(args.train_paths)
    train_file_paths.extend(args.generator_train_paths)


    set_seed(args.seed)

    train_data, train_label, _ = load_and_combine_npz(
        file_paths=train_file_paths,
        file_specific_limit={},
        shuffle=True, 
        include_csv_names=False
    )
    train_dataset = TensorDataset(torch.cat([train_data], dim=0), torch.cat([train_label], dim=0))

    npz_result_val = args.npz_result_val
    npz_result_test = args.npz_result_test


    val_data, val_label, val_csv_names = load_and_combine_npz(file_paths=[npz_result_val], include_csv_names=True)
    val_dataset = PatientDataset(val_data, val_label, val_csv_names)

    test_data, test_label, test_csv_names = load_and_combine_npz(file_paths=[npz_result_test], include_csv_names=True)
    test_dataset = PatientDataset(test_data, test_label, test_csv_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discriminative_score_metrics(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        best_model_path=args.best_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        pretrain_path = args.pretrain_path
    )