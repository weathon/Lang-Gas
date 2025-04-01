import numpy as np

class BinaryConfusion:
    def __init__(self, backend="torch"):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        assert backend in ["torch", "numpy"], "Invalid backend"
        if backend == "torch":
            import torch
            self.torch = torch
        self.backend = backend

    def update(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert y_true.shape == y_pred.shape
        if self.backend == "torch":
            self.tp += self.torch.sum((y_true == 1) & (y_pred == 1))
            self.fn += self.torch.sum((y_true == 1) & (y_pred == 0))
            self.fp += self.torch.sum((y_true == 0) & (y_pred == 1))
            self.tn += self.torch.sum((y_true == 0) & (y_pred == 0))
        elif self.backend == "numpy":
            self.tp += np.sum((y_true == 1) & (y_pred == 1))
            self.fn += np.sum((y_true == 1) & (y_pred == 0))
            self.fp += np.sum((y_true == 0) & (y_pred == 1))
            self.tn += np.sum((y_true == 0) & (y_pred == 0))
        else:
            raise ValueError("Invalid backend")
            
            

    def get_f1(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
    
    def get_iou(self):
        return self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else 0