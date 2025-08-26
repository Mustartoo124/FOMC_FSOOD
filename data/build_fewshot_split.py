import yaml
import json
import random
import os
from models.masking import apply_shot_masking  # Import masking function
from tqdm import tqdm
from class_names import class_names

with open('configs/splits.yaml', 'r') as f:
    splits = yaml.safe_load(f)

base_classes = splits['base']
novel_classes = splits['novel']
shots_list = splits['fewshot']['shots']
seeds = splits['fewshot']['seeds']

ANN_JSON = 'data/dota/annotations.json'

def build_splits():
    with open(ANN_JSON, 'r') as f:
        data = json.load(f)

    for seed in range(seeds):
        random.seed(seed)
        for k in shots_list:
            fewshot_ann = []
            for cat in novel_classes:
                cat_anns = [ann for ann in data['annotations'] if ann['category_id'] == class_names.index(cat)]
                selected = random.sample(cat_anns, min(k * 10, len(cat_anns)))  # ~10 instances per shot as in paper
                fewshot_ann.extend(selected)
            # Combine with base anns
            base_anns = [ann for ann in data['annotations'] if ann['category_id'] in [class_names.index(c) for c in base_classes]]
            combined = base_anns + fewshot_ann

            # Apply masking to images: blur non-selected in novel images
            for img in tqdm(data['images']):
                img_anns = [ann for ann in fewshot_ann if ann['image_id'] == img['id']]
                if img_anns:  # Novel image
                    apply_shot_masking(img['file_name'], img_anns, threshold=0.5)  # Mask others

            os.makedirs(f'data/splits/seed{seed}_k{k}', exist_ok=True)
            with open(f'data/splits/seed{seed}_k{k}/annotations.json', 'w') as f:
                json.dump({'annotations': combined, 'images': data['images'], 'categories': data['categories']}, f)

if __name__ == '__main__':
    build_splits()
