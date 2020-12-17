import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def mosei_metrics(y_true, y_pred):
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])

    y_true_bin = y_true[non_zeros] > 0
    y_pred_bin = y_pred[non_zeros] > 0

    bi_acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_pred.reshape(-1), y_true.reshape(-1))[0][1]
    return bi_acc, f1, mae, corr
