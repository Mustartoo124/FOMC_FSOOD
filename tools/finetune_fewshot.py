import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.builder import build_model
from tools.utils import DOTA_Dataset, focal_loss, smooth_l1  # Custom dataset and losses
from tqdm import tqdm

def train_base(config_path):
    model, cfg = build_model(config_path)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = DOTA_Dataset(cfg['dataset']['root_dir'], base_only=True)  # Only base classes
    loader = DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'])

    optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'], momentum=cfg['training']['momentum'], weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['step_size'], gamma=cfg['training']['gamma'])

    for epoch in range(cfg['training']['epochs']):
        for imgs, bboxes, labels in tqdm(loader):
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            loss = model(imgs, bboxes, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save(model.state_dict(), f"{cfg['checkpoint_dir']}/epoch{epoch}.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_base.yaml')
    args = parser.parse_args()
    train_base(args.config)