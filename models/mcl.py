import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ProjectionEncoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, 128, 1)  # Embedding dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, x.size()[2:])  # Global pool for proposals
        x = x.view(x.size(0), -1)
        return F.normalize(x, dim=1)

class MCL(nn.Module):
    def __init__(self, embed_dim=128, memory_size=8192, tau=0.07, theta=0.5):
        super().__init__()
        self.projector = ProjectionEncoder()
        self.memory_bank = deque(maxlen=memory_size)  # Queue of (embed, iou, label)
        self.tau = tau
        self.theta = theta

    def forward(self, features, ious, labels):
        embeds = self.projector(features)  # B x 128
        # Enqueue current batch (high IoU only)
        for e, i, l in zip(embeds, ious, labels):
            if i > self.theta:
                self.memory_bank.append((e, i, l))

        # Compute contrastive loss
        loss = 0
        for idx, (q, iou_q, lbl_q) in enumerate(zip(embeds, ious, labels)):
            if iou_q <= self.theta:
                continue  # Only centered proposals
            positives = [p[0] for p in self.memory_bank if p[2] == lbl_q and p != q]
            negatives = [p[0] for p in self.memory_bank if p[2] != lbl_q]
            if not positives:
                continue
            sim_pos = torch.exp(torch.matmul(q, torch.stack(positives).T) / self.tau)
            sim_neg = torch.exp(torch.matmul(q, torch.stack(negatives).T) / self.tau)
            loss += -torch.log(sim_pos.mean() / (sim_pos.mean() + sim_neg.sum() + 1e-6)) * iou_q  # Weighted by IoU

        return loss / len(embeds) if len(embeds) > 0 else 0
