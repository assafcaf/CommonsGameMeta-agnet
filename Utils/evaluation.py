import torch
import numpy as np

def compute_accuracy(model, loader, device):
    accuracy = []
    for x, y in loader:
        x = x.to(device)
        y_pred = torch.flatten(model(x)[0]).type(torch.FloatTensor)
        y = torch.flatten(y)
        correct = (y_pred == y).type(torch.FloatTensor)
        accuracy.append(correct.mean())
    return np.mean(accuracy)

