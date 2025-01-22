import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from util.loader import apply_gaussian_blur

def compute_mask_with_threshold_and_blur(mask, threshold=0.1, record=False, name=None):

    # Apply Gaussian blur
    mask = apply_gaussian_blur(mask, kernel_size=5, sigma=2.0, threshold_of_blur=0.8)
    # Apply threshold to get binary mask
    mask = (mask > threshold).float()
    # Save mask if record is True
    if record:
        if not os.path.exists('mask_smooth'):
            os.makedirs('mask_smooth')
        # Convert mask from [1,1,H,W] to [H,W] before saving
        mask_2d = mask.squeeze().cpu().numpy()
        plt.imsave(f'mask_smooth/{name}.png', mask_2d, cmap='gray')
    return mask

def compute_auroc(loader, out_path, batch_idx=0):
    y_true = []
    y_scores = []
    
    for i, ref_img_dict in enumerate(loader):
        fname = os.path.basename(ref_img_dict['category'][batch_idx]) + '_' + ref_img_dict['name'][batch_idx]
        
        pred_mask_path = os.path.join(out_path, 'mask', fname)
        pred_mask = Image.open(pred_mask_path).convert('L')
        pred_mask = transforms.ToTensor()(pred_mask)
        
        if 'mask' in ref_img_dict:
            gt_mask = ref_img_dict['mask'][batch_idx]
        else:
            gt_mask = torch.zeros_like(pred_mask)
            
        # 二值化ground truth
        gt_mask = (gt_mask > 0.5).float()
        pred_mask_flat = compute_mask_with_threshold_and_blur(pred_mask, threshold=0, record=True, name= ref_img_dict['category'][batch_idx] + '_' + ref_img_dict['name'][batch_idx])
        pred_mask_flat = pred_mask_flat.flatten()
        gt_mask_flat = gt_mask.flatten()
        
        y_true.extend(gt_mask_flat.cpu().numpy())
        y_scores.extend(pred_mask_flat.cpu().numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()
    
    return auroc


if __name__ == '__main__':
    pass