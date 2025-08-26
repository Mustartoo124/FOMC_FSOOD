import cv2
import numpy as np

def apply_shot_masking(img_path, selected_anns, threshold=0.5):
    img = cv2.imread(img_path)
    mask = np.zeros_like(img)
    for ann in selected_anns:
        poly = np.array(ann['bbox']).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [poly], (255, 255, 255))  # White for selected

    # Blur where mask == 0 (non-selected)
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    img = np.where(mask == 0, blurred, img)
    cv2.imwrite(img_path.replace('.png', '_masked.png'), img)  # Save masked version
