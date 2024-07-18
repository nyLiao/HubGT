import torch
import torch.nn as nn


class F1Calculator(object):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.TP = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)

    def reset(self):
        self.TP.zero_()
        self.FP.zero_()
        self.FN.zero_()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true = nn.functional.one_hot(y_true, num_classes=self.num_classes)
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            y_pred = nn.functional.one_hot(y_pred, num_classes=self.num_classes)
        self.TP += (y_true * y_pred).sum(dim=0).cpu()
        self.FP += ((1 - y_true) * y_pred).sum(dim=0).cpu()
        self.FN += (y_true * (1 - y_pred)).sum(dim=0).cpu()

    def compute(self, average: str='micro'):
        eps = 1e-10
        if average == 'micro':
            # For multi-class classification, F1 micro is equivalent to accuracy
            f1 = 2 * self.TP.float().sum() / (2 * self.TP.sum() + self.FP.sum() + self.FN.sum() + eps)
            # return f1.item()
            return [('f1i', f1.item(), lambda x: format(x*100, '.3f'))]
        elif average == 'macro':
            f1 = 2 * self.TP.float() / (2 * self.TP + self.FP + self.FN + eps)
            # return f1.mean().item()
            return [('f1a', f1.mean().item(), lambda x: format(x*100, '.3f'))]
        else:
            raise ValueError('average must be "micro" or "macro"')

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
