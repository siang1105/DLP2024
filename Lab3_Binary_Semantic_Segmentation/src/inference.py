import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score, dice_loss
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(args):
    data_path = args.data_path
    model_path = args.model
    batch_size = args.batch_size

    # load data
    data_loader = torch.utils.data.DataLoader(load_dataset(data_path, "test"), batch_size=batch_size, shuffle=False)
    data_loader = tqdm(data_loader, desc="[PREDICT]", leave=False)

    if "resnet34_unet" in model_path:
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    elif "unet" in model_path:
        model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Unknown model type in the model path")

    # load model
    model.load_state_dict(torch.load(f"../saved_models/{args.model}", map_location=device, weights_only=True))

    model.eval()
    dice_scores = []
    test_loss = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)

            dice_scores.append(dice_score(outputs, masks).item())
            test_loss.append(criterion(outputs, masks).item() + dice_loss(outputs, masks).item())
            data_loader.set_postfix(loss=np.mean(test_loss), dice=np.mean(dice_scores))

    avg_dice = np.mean(dice_scores)
    avg_loss = np.mean(test_loss)

    print(f'[Testing] Dice Score: {avg_dice:.4f}, Loss: {avg_loss:.4f}')
    return avg_loss, avg_dice

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='unet.pth', choices=["unet.pth", "resnet34_unet.pth"])
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    predict(args)



# python3 inference.py --data_path ../dataset/oxford-iiit-pet --batch_size 16 --model unet
# python3 inference.py --data_path ../dataset/oxford-iiit-pet --batch_size 16 --model resnet34_unet.pth

# python3 inference.py --data_path ../dataset/oxford-iiit-pet --batch_size 16 --model best_model_unet_epoch126.pth