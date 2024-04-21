import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score

def classify(model, device, dataset,epoch, batch_size=128):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    y_trues = np.empty((0, len(dataset.CLASSES)))
    y_preds = np.empty((0, len(dataset.CLASSES)))

    y_scores_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            y_pred = torch.sigmoid(y_hat).cpu().numpy().round()
            y = y.cpu()

            y_preds = np.concatenate((y_preds, y_pred), axis=0)
            y_trues = np.concatenate((y_trues, y), axis=0)

            y_scores = torch.sigmoid(y_hat).cpu().numpy()
            y_scores_list.append(y_scores)
    y_scores_all = np.concatenate(y_scores_list, axis=0)

    df = pd.DataFrame(y_scores_all, columns=dataset.CLASSES)

    # Lưu DataFrame vào tập tin CSV
    df.to_csv(f'./ptbxl/checkpoints/{epoch}_y_scores.csv', index=False, header=False)

    return y_trues, y_preds

def get_f1(y_trues, y_preds):
    f1 = []
    for j in range(y_trues.shape[1]):
        f1.append(f1_score(y_trues[:, j], y_preds[:, j]))
    return np.array(f1)

def get_auprc(y_trues, y_scores):
    auprc = []
    for j in range(y_trues.shape[1]):
        p, r, thresholds = precision_recall_curve(y_trues[:, j], y_scores[:, j])
        auprc.append(auc(r, p))

    return np.array(auprc)

