import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from util.loader import apply_gaussian_blur

def compute_mask_with_threshold_and_blur(mask, threshold=0.1):
    mask = (mask < threshold).float()  # Changed > to < to match instructions

    mask = apply_gaussian_blur(mask, kernel_size=5, sigma=2.0, threshold_of_blur=0.5)
    return mask

def compute_auroc(loader, out_path, batch_idx=0):
    # Initialize lists to store true labels and predicted scores
    y_true = []
    y_scores = []
    
    # Process each image in the loader
    for i, ref_img_dict in enumerate(loader):
        fname = os.path.basename(ref_img_dict['category'][batch_idx]) + '_' + ref_img_dict['name'][batch_idx]
        
        # 读取 PNG 格式的 mask
        pred_mask_path = os.path.join(out_path, 'mask', fname)
        pred_mask = Image.open(pred_mask_path).convert('L')  # 转换为灰度图
        pred_mask = transforms.ToTensor()(pred_mask)  # 转换为tensor [0,1]
        if 'mask' in ref_img_dict:
            gt_mask = ref_img_dict['mask'][batch_idx]
        else:
            gt_mask = torch.zeros_like(pred_mask)
        
        # Flatten masks to 1D arrays
        pred_mask_flat = pred_mask.flatten()
        gt_mask_flat = gt_mask.flatten()
        print("pred_mask_flat: ", pred_mask_flat)
        print("pred_mask_flat.shape: ", pred_mask_flat.shape)
        print("gt_mask_flat: ", gt_mask_flat)
        print("gt_mask_flat.shape: ", gt_mask_flat.shape)
        exit()
        # Add to lists
        y_true.extend(gt_mask_flat.cpu().numpy())
        y_scores.extend(pred_mask_flat.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate ROC curve and AUROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)
    
    return auroc


if __name__ == '__main__':
    pass