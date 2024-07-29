import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def dice_score(pred_mask, gt_mask, eps = 1e-6):
    # Set predicted mask values ​​greater than 0.5 to 1.0, otherwise to 0.0
    pred_mask = torch.where(pred_mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))

    # Count the number of parts where the predicted mask and the true mask are equal, using epsilon (eps) to avoid numerical problems
    intersection = (abs(pred_mask - gt_mask) < eps).sum()

    # Calculate the total number of all pixels in the predicted mask and the real mask
    total_pix = pred_mask.reshape(-1).shape[0] + gt_mask.reshape(-1).shape[0]

    # Calculate Dice score, the formula is (2 * intersection) / total number of pixels
    dice_score = 2 * intersection / total_pix
    return dice_score

def dice_loss(pred_mask, gt_mask, eps=1e-8):
    intersection = torch.sum(gt_mask * pred_mask) + eps
    total_pix = torch.sum(gt_mask) + torch.sum(pred_mask) + eps
    # Calculate Dice loss, the formula is 1 - (2 * intersection / total number of pixels)
    loss = 1 - (2 * intersection / total_pix)
    return loss

def visualize_prediction(images, masks, outputs, idx, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))
    axes[0].set_title('Input Image')
    axes[1].imshow(masks[idx][0], cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(outputs[idx][0] > 0.5, cmap='gray')
    axes[2].set_title('Prediction')
    plt.savefig(os.path.join(save_path, f'prediction_{idx}.png'))