import os
import torch
from models.builder import build_model
from tools.utils import DOTA_Dataset, decode_obb
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('data/DOTA_devkit')
from data.DOTA_devkit.dota_evaluation_task1 import task1_eval  # Import eval function

def evaluate(model_path, split='test'):
    model, cfg = build_model('configs/train_base.yaml')  # Base config, adjust
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = DOTA_Dataset(cfg['dataset']['root_dir'], split=split)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    predictions = []
    for imgs, _, _ in tqdm(loader):
        imgs = imgs.to(device)
        cls, reg, anchors = model(imgs, mode='infer')
        # Post-process: NMS for rotated boxes, decode OBBs
        pred_bboxes = decode_obb(anchors, reg, cls)  # Implement in utils
        predictions.append(pred_bboxes)

    # Save predictions in DOTA format (txt per image)
    save_dir = 'eval/preds/'
    os.makedirs(save_dir, exist_ok=True)
    for i, pred in enumerate(predictions):
        with open(os.path.join(save_dir, f'img{i}.txt'), 'w') as f:
            for bbox in pred:
                x1,y1,x2,y2,x3,y3,x4,y4,score,cls = bbox
                f.write(f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {cls} {score}\n')

    # Run DOTA eval
    gt_dir = 'data/dota/labelTxt/'  # Ground truth
    mAP = task1_eval(save_dir, gt_dir)  # Returns mAP dict
    print(f"mAP: {mAP['mAP']}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    evaluate(args.model_path, args.split)