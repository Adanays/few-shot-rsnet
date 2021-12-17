import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
act = torch.nn.ReLU()
from wrn_mixup_model import wrn28_10
from scipy.spatial.distance import cdist


class SimilarityHead(nn.Module):
    def __init__(self, n_feat=640):
        super(SimilarityHead, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat * 2),
            nn.BatchNorm1d(n_feat * 2),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(n_feat, n_feat),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Linear(n_feat, 2)

        self.diff_dim = n_feat

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        diff_feat = x

        x = self.fc3(x)
        x = self.fc4(x)

        return x, diff_feat

class RSN_Head(nn.Module):
    def __init__(self, n_feat=320):
        super(RSN_Head, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat * 2),
            nn.BatchNorm1d(n_feat * 2),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.fc1(x)
        sim_feat = self.fc2(x)
        diff_feat = self.fc3(x)

        return sim_feat, diff_feat

def Sample_pairing(x, dist_type='euclidean'):
    '''
    x:     (bacth_size, c, h, w)
    label: (batch_size, )
    return (bacth_size*3, c, h, w)
    '''
    b = x.size(0)
    x = x.cpu().detach().numpy()
    if dist_type == 'euclidean':
        dists = cdist(x, x, metric='euclidean')
        for i in range(b):
            dists[i, i] = 999
        min_indexs = np.argmin(dists, axis=0)

    return min_indexs

class RSNet(nn.Module):
    def __init__(self, num_classes=200 , loss_type = 'dist', pm=False):
        super(RSNet, self).__init__()
        self.backbone = wrn28_10(num_classes=num_classes, loss_type=loss_type)
        self.rsn_head = RSN_Head(n_feat=320)

    def Feature_mixup(self, sim_feat, diff_feat, target, min_Pairs):
        min_Pairs = torch.tensor(min_Pairs, device='cuda')
        min_sim = torch.index_select(sim_feat, dim=0, index=min_Pairs)
        min_diff = torch.index_select(diff_feat, dim=0, index=min_Pairs)

        mix_feat2 = torch.cat((sim_feat, min_diff), dim=1)
        mix_label2 = torch.index_select(target, dim=0, index=min_Pairs)
        mix_feat3 = torch.cat((min_sim, diff_feat), dim=1)
        mix_label3 = target

        mixed_feat = torch.cat((mix_feat2, mix_feat3), dim=0)
        mixed_label = torch.cat((mix_label2, mix_label3))

        return mixed_feat, mixed_label

    def forward(self, x, target = None, mixup=False, mixup_hidden=True, mixup_alpha=None , lam = 0.4):
        '''
            the [:,1] is for similarity, and [:,0] dissimilarity.
        '''
        # if self.training:
        if target is not None:
            features, outputs , target_a , target_b, new_target = self.backbone(x, target, mixup_hidden= mixup_hidden,
                                                                                mixup_alpha = mixup_alpha , lam = lam)

            min_Pairs = Sample_pairing(features)  # (index(0), arr[0])
            sim_feat, diff_feat = self.rsn_head(features)

            mixed_feat, mixed_label = self.Feature_mixup(sim_feat, diff_feat, new_target, min_Pairs)   # mix_feat.size(0) = features.size(0) * 4
            mix_cls_scores = self.backbone.linear(mixed_feat)
            min_Pairs = torch.tensor(min_Pairs, device='cuda')

            return features, outputs , target_a , target_b, mix_cls_scores, mixed_label, sim_feat, diff_feat, min_Pairs
        else:
            features, outputs = self.backbone(x)
            sim_feat, diff_feat = self.rsn_head(features)
            features = torch.cat((sim_feat, diff_feat), dim=1)
            return features, outputs
