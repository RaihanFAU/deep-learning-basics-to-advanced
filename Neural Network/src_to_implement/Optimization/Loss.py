import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):

        self.y = label_tensor
        self.y_hat = prediction_tensor
        #print(np.finfo(float).eps)

        eps = np.finfo(float).eps
        safe_y_hat = self.y_hat + eps
        loss_terms = np.where(self.y == 1, -np.log(safe_y_hat), 0)
        return loss_terms.sum()

    def backward(self, label_tensor):
        eps = np.finfo(float).eps
        numerator = -label_tensor
        denominator = self.y_hat + eps
        return numerator / denominator
