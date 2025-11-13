import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math
from models.module.MFIM import MFIM
from models.module.PRM import PRM

def pdist(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is None:
        y = x
    y_t = y.transpose(0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return dist

class MPRe(nn.Module):
    def __init__(self, way=5, shots=[5,15], resnet=False):
        super().__init__()
        self.resnet = resnet
        if self.resnet:
            self.num_channel = 640 
            self.dim = 640 * 25
            self.feature_extractor = ResNet.resnet12()
        else:
            self.num_channel = 64
            self.dim = 64 * 25
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.way = way
        self.shots = shots
        self.shot, self.query_shot = shots[0], shots[1]

        self.mfim = MFIM(self.resnet, self.num_channel)
        self.prm_h = PRM(self.resnet, self.num_channel)
        self.prm_m = PRM(self.resnet, self.num_channel)
        self.scale_h = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_m = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_feature_vector(self, inp):
        f_h, f_m = self.feature_extractor(inp)
        return f_h, f_m

    def get_neg_l2_dist(self, inp, way, shot, query_shot):
        f_h, f_m = self.get_feature_vector(inp)
        f_refine_h, f_refine_m = self.mfim(f_h, f_m)
        proto_h, query_h = self.prm_h(f_refine_h, way, shot)
        proto_m, query_m = self.prm_m(f_refine_m, way, shot)
        neg_l2_h = pdist(query_h, proto_h).neg()
        neg_l2_m = pdist(query_m, proto_m).neg()
        return neg_l2_h, neg_l2_m

    def meta_test(self, inp, way, shot, query_shot):
        neg_l2_dist_h, neg_l2_dist_m = self.get_neg_l2_dist(inp, way, shot, query_shot)
        neg_l2_dist_all = neg_l2_dist_h + neg_l2_dist_m
        _, max_index = torch.max(neg_l2_dist_all, 1)
        return max_index

    def forward(self, inp):
        neg_l2_dist_h, neg_l2_dist_m = self.get_neg_l2_dist(inp, self.way, self.shots[0], self.shots[1])
        logits_h = neg_l2_dist_h / self.dim * self.scale_h
        logits_m = neg_l2_dist_m / self.dim * self.scale_m
        log_prediction_h = F.log_softmax(logits_h, dim=1)
        log_prediction_m = F.log_softmax(logits_m, dim=1)
        return log_prediction_h, log_prediction_m
