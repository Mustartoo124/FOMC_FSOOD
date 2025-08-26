import yaml
from s2anet_mcl_head import S2ANetMCL
from data.class_names import num_classes
import torch

def build_model(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model = S2ANetMCL(num_classes)
    if 'load_base' in cfg['model']:
        model.load_state_dict(torch.load(cfg['model']['load_base']), strict=False)
    if cfg['training'].get('freeze_backbone', False):
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model, cfg
