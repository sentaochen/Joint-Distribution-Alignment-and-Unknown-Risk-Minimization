import torch
import torch.nn as nn
import numpy as np
from utils import globalvar as gl

def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = torch.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[int(true)]][pred_label_to_id[int(pred)]] += 1.0
    return confusion_matrix

def test_OSDA(total_label_t, total_pred_t, known_num_class):
    max_target_label = int(torch.max(total_label_t)+1)
    m = extended_confusion_matrix(total_label_t, total_pred_t, true_labels=list(range(max_target_label)), pred_labels=list(range(known_num_class+1)))
    cm = m
    cm = cm.long()/ torch.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(known_num_class)]) / known_num_class
    acc_unknown = sum([cm[i][known_num_class] for i in range(known_num_class, int(torch.max(total_label_t)+1))])
    acc_os = (acc_os_star * (known_num_class) + acc_unknown) / (known_num_class+1)
    acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)
    return acc_os*100, acc_os_star*100, acc_unknown*100, acc_hos*100


def test(loader, model):
    DEVICE = gl.get_value('DEVICE')
    model.eval()
    size = len(loader.dataset)
    start_test = True
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model.get_fc_features(inputs)
            if start_test:
                all_outputs = outputs.float()
                all_labels = labels.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.float()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)
        all_preds = torch.max(all_outputs, 1)[1]
        correct = torch.sum(all_preds == all_labels.data).item()
        acc = 100.0 * (float)(correct) / size
    return acc, all_preds, all_labels

def predict(model, loader):
    DEVICE = gl.get_value('DEVICE')
    model.eval()
    labels = []
    pivot = 2.0
    with torch.no_grad():
        # DO NOT use the true label, just generate the pseudo labels
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            outputs = model.get_fc_features(inputs)
            preds = torch.max(outputs, 1)[1].tolist()
            labels += preds
    return labels
