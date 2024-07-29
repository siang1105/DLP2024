import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import jaccard_score
from torch.utils.tensorboard import SummaryWriter

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score, dice_loss
import evaluate

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# train
def train_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    dice_scores = []
    running_loss = []

    train_loader = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)

    for batch in train_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        outputs = model(images)
        
        loss = criterion(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice_scores.append(dice_score(outputs, masks).item())

        running_loss.append(loss.item())

        # Dynamically update progress bar description
        train_loader.set_postfix(loss=np.mean(running_loss), dice=np.mean(dice_scores))

    epoch_loss = np.mean(running_loss)
    epoch_dice = np.mean(dice_scores)

    print(f"Train Loss: {epoch_loss:.4f}, Train Dice: {epoch_dice:.4f}")

    # write loss and dice score to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Dice/train', epoch_dice, epoch)

    return epoch_loss, epoch_dice

# train
def train(args):
    data_path = args.data_path
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_type = args.model

    train_loader = torch.utils.data.DataLoader(load_dataset(data_path, "train"), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(load_dataset(data_path, "valid"), batch_size=batch_size, shuffle=False)

    if model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif model_type == "resnet34_unet":
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Unknown model type: {}".format(model_type))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_dice = 0.9

    # SummaryWriter initialize TensorBoard SummaryWriter
    writer = SummaryWriter()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        val_dice, val_loss = evaluate.evaluate(model, val_loader, criterion, device)

        # write valid loss and dice score to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)

        # print valid result
        tqdm.write(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            model_path = os.path.join('../saved_models', f"{model_type}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

    # close SummaryWriter
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet or ResNet34_UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'resnet34_unet'], help='model type to use')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
