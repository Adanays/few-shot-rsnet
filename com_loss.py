import torch.nn as nn
import torch

class com_Loss(nn.Module):
    """Common feature loss.

    Args:
        x (n, m): feature matrix of images.
        min_index (list): min dist index.
        delta (float): weight.
    """
    def __init__(self, delta=1e-6, use_gpu=True):
        super(com_Loss, self).__init__()
        self.delta = delta
        self.us_gpu = use_gpu

    def forward(self, x, min_index):
        min_com = torch.index_select(x, dim=0, index=min_index)
        if self.us_gpu:
            x = x.cuda()
            min_com = min_com.cuda()
        feat = x + self.delta - min_com
        L_com = torch.norm(feat[:,None], dim=2, p=2)
        L_com = torch.mean(L_com)
        return L_com