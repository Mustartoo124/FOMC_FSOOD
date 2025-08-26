import os
import json
from tqdm import tqdm
import cv2
from scipy.io import loadmat  # If needed for mat files, but DOTA is txt
import numpy as np

DOTA_ROOT = 'data/dota/'
IMAGE_DIR = os.path.join(DOTA_ROOT, 'images')
ANN_DIR = os.path.join(DOTA_ROOT, 'labelTxt')
OUTPUT_JSON = os.path.join(DOTA_ROOT, 'annotations.json')

class_names = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

def prepare_dota():
    if not os.path.exists(DOTA_ROOT):
        os.makedirs(DOTA_ROOT)
        print("Download DOTA-v1.0 from https://captain-whu.github.io/DOTA/dataset.html and extract to data/dota/")
        return

    annotations = []
    image_id = 0
    ann_id = 0
    for ann_file in tqdm(os.listdir(ANN_DIR)):
        if ann_file.endswith('.txt'):
            img_file = ann_file.replace('.txt', '.png')  # Assuming png
            img_path = os.path.join(IMAGE_DIR, img_file)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            with open(os.path.join(ANN_DIR, ann_file), 'r') as f:
                lines = f.readlines()[2:]  # Skip header
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    x1, y1, x2, y2, x3, y3, x4, y4, cls, diff = parts
                    cls_id = class_names.index(cls)
                    poly = [float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
                    # Convert poly to OBB (center, w, h, angle) - implement conversion if needed
                    annotations.append({
                        'image_id': image_id,
                        'id': ann_id,
                        'category_id': cls_id,
                        'bbox': poly,  # Keep as poly for now
                        'area': np.abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1)) / 2,
                        'iscrowd': 0
                    })
                    ann_id += 1
            image_id += 1

    with open(OUTPUT_JSON, 'w') as f:
        json.dump({'images': [...], 'annotations': annotations, 'categories': [{'id': i, 'name': n} for i, n in enumerate(class_names)]}, f)  # Fill images list similarly

if __name__ == '__main__':
    prepare_dota()
