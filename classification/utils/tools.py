import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from collections import defaultdict

def evaluate_model(model, dataloader, criterion, device, threshold=0.5):
    """
    A unified evaluation function suitable for validation and test datasets.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device (CPU or GPU) to run the model on.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval()
    y_true, y_pred, y_score = [], [], []
    all_csv_names = []
    total_loss = 0.0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for data, labels, csv_names in progress:
            data = data.to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.squeeze(-1), labels)
            total_loss += loss.item() * data.size(0)

            # Calculate probabilities and predictions
            probabilities = torch.sigmoid(outputs.squeeze(-1))
            preds = (probabilities > threshold).int()

            # Collect results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())
            all_csv_names.extend(csv_names)

    # Aggregate results by patient
    patient_results = defaultdict(lambda: {'scores': [], 'true_label': None})
    for csv_name, true_label, score in zip(all_csv_names, y_true, y_score):
        patient_results[csv_name]['true_label'] = true_label
        patient_results[csv_name]['scores'].append(score)

    # Calculate patient-level metrics
    patient_true = []
    patient_scores = []
    for csv_name, data in patient_results.items():
        patient_true.append(data['true_label'])
        patient_scores.append(np.mean(data['scores']))
    
    patient_pred = [1 if score >= threshold else 0 for score in patient_scores]

    # Calculate evaluation metrics
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

def print_metrics(metrics, dataset_name, verbose=False, patient_true=None, patient_pred=None):
    """
    Print evaluation metrics.

    Args:
        metrics (dict): A dictionary containing evaluation metrics.
        dataset_name (str): Name of the dataset (e.g., 'Validation', 'Test').
        verbose (bool, optional): Whether to print detailed classification report and confusion matrix. Defaults to False.
        patient_true (list, optional): True labels at the patient level. Defaults to None.
        patient_pred (list, optional): Predicted labels at the patient level. Defaults to None.
    """
    print(f"\n{dataset_name} Evaluation Results:")
    print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall: {metrics['recall']:.4f}")
    print(f"- Macro F1: {metrics['classification_report']['macro avg']['f1-score']:.4f}")

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