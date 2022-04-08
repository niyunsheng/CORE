from torch import nn
import torch

class ConsistencyCos(nn.Module):
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)

        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available():
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        return loss

class ConsistencyL2(nn.Module):
    def __init__(self):
        super(ConsistencyL2, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        loss = self.mse_fn(feat_0, feat_1)
        return loss

class ConsistencyL1(nn.Module):
    def __init__(self):
        super(ConsistencyL1, self).__init__()
        self.L1_fn = nn.L1Loss()

    def forward(self, feat):
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        loss = self.L1_fn(feat_0, feat_1)
        return loss