import torch.nn as nn
import torch
from torchvision.models import resnet50
from tools.utils import build_fpn, focal_loss, compute_iou  # Assume utils has FPN, focal_loss, and compute_iou implementations
from mcl import MCL

class FAM(nn.Module):
    def __init__(self):
        super().__init__()
        # Anchor Refinement Network (ARN): Generate rotated anchors
        self.arn = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 5, 1))  # 5 for OBB params
        # Alignment Conv Layer (ACL)
        self.acl = nn.Conv2d(256, 256, 3, padding=1)  # Simplified

    def forward(self, x):
        anchors = self.arn(x)
        aligned = self.acl(x)  # Align features to anchors
        return aligned, anchors

class ODM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Adaptive Rotation Filtering (ARF)
        self.arf_os = nn.Conv2d(256, num_classes, 1)  # Orientation-sensitive for reg
        self.arf_oi = nn.Conv2d(256, num_classes, 1)  # Orientation-invariant for cls

    def forward(self, x):
        cls = self.arf_oi(x)  # Class scores
        reg = self.arf_os(x)  # OBB regression: dx, dy, dw, dh, dtheta
        return cls, reg

class S2ANetMCL(nn.Module):
    def __init__(self, num_classes, embed_dim=128, lambda_mcl=1.0):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.lambda_mcl = lambda_mcl
        self.fpn = build_fpn(self.backbone)  # Custom FPN from utils
        self.fam = FAM()
        self.odm = ODM(num_classes)
        self.mcl = MCL(embed_dim)

    def forward(self, x, gt_bboxes=None, gt_labels=None, mode='train'):
        feats = self.fpn(self.backbone(x))
        aligned, anchors = self.fam(feats)
        cls, reg = self.odm(aligned)
        if mode == 'train':
            # Compute losses
            L_cls = focal_loss(cls, gt_labels)  # Implement in utils
            criterion_reg = nn.SmoothL1Loss()
            L_reg = criterion_reg(reg, gt_bboxes)
            features_oi = cls  # Example: Use class scores as orientation-invariant features
            ious = compute_iou(anchors, gt_bboxes)  # Custom IoU for rotated
            L_mcl = self.mcl(features_oi, gt_labels)  # Compute MCL loss using MCL module
            return L_cls + L_reg + self.lambda_mcl * L_mcl
            
        else:
            return cls, reg, anchors  # For inference
 
