import torch.nn as nn
import torch

class pri_Loss(nn.Module):
    """Private feature loss.

    Args:
        x (n, m): feature matrix of images.
        min_index (list): min dist index.
        delta (float): weight.
    """
    def __init__(self, delta=1e-6, use_gpu=True):
        super(pri_Loss, self).__init__()
        self.delta = delta
        self.us_gpu = use_gpu

    def forward(self, x, min_index):
        min_pri = torch.index_select(x, dim=0, index=min_index)
        if self.us_gpu:
            x = x.cuda()
            min_pri = min_pri.cuda()
        feat = x + self.delta - min_pri
        L_pri = torch.norm(feat[:,None], dim=2, p=2)
        L_pri = torch.pow(torch.mean(L_pri), -1)
        return L_pri