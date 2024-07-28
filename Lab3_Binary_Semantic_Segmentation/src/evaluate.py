import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import dice_score, dice_loss
import numpy as np

def evaluate(net, data_loader, criterion, device):
    net.eval()
    dice_scores = []
    val_loss = []

    data_loader = tqdm(data_loader, desc="[EVALUATE]", leave=False)

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = net(images)
            dice_scores.append(dice_score(outputs, masks).item())
            val_loss.append(criterion(outputs, masks).item() + dice_loss(outputs, masks).item())

            # Dynamically update progress bar description
            data_loader.set_postfix(loss=np.mean(val_loss), dice=np.mean(dice_scores))

    avg_dice = np.mean(dice_scores)
    avg_loss = np.mean(val_loss)

    print(f'Evaluation Dice Score: {avg_dice:.4f}, Loss: {avg_loss:.4f}')
    return avg_dice, avg_loss


