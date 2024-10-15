import time
from pprint import pprint

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

from src.dataset_loader import data_loader
from src.model import ResNet, ResidualBlock


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
                predicted_v.append(pred.cpu().detach().numpy())
                labeled_v.append(lab.cpu().detach().numpy())
                results_matrix[lab, pred] += 1
            del images, labels, outputs
    acc = accuracy_score(labeled_v, predicted_v)
    f1_scores = f1_score(labeled_v, predicted_v, average='micro')
    prec = precision_score(labeled_v, predicted_v, average='micro')
    rec = recall_score(labeled_v, predicted_v, average='micro')
    classification = classification_report(labeled_v, predicted_v)
    confusion_matrix_res = confusion_matrix(labeled_v, predicted_v)
    return {"acc": acc, "f1": f1_scores, "prec": prec, "rec": rec, "classification": classification,
            "conf_matrix": confusion_matrix_res, "time_all": time_all}


if __name__ == "__main__":
    MODEL_PATH_IN = "/home/kirrog/projects/FQWB/model/aug_4_block_reg_group/l1_0.0001_l2_1e-05_wd_1e-08/result_acc_0.807900.bin"
    model = ResNet(ResidualBlock, [3, 1, 1, 3])
    model.load_state_dict(torch.load(MODEL_PATH_IN))
    model.eval()
    model = model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              test=True)
    metrics = calc_metrics(model, test_loader, device)
    pprint(metrics)
