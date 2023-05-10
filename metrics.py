import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from src.model import ResNet


def calc_metrics(model: ResNet, dataset_loader, device):
    results_matrix = np.zeros((model.num_classes, model.num_classes))
    predicted_v = []
    labeled_v = []
    time_all = 0
    with torch.no_grad():
        for images, labels in dataset_loader:
            time_local = time.time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            time_all += time.time() - time_local
            for pred, lab in zip(predicted, labels):
                predicted_v.append(pred)
                labeled_v.append(lab)
                results_matrix[lab, pred] += 1
            del images, labels, outputs
    acc = accuracy_score(labeled_v, predicted_v)
    f1_scores = f1_score(labeled_v, predicted_v)
    prec = precision_score(labeled_v, predicted_v)
    rec = recall_score(labeled_v, predicted_v)
    classification = classification_report(labeled_v, predicted_v)
    confusion_matrix_res = confusion_matrix(labeled_v, predicted_v)
    return acc, f1_scores, prec, rec, classification, confusion_matrix_res, time_all

