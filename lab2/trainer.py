import os
import warnings
import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet
from Dataloader import MIBCI2aDataset

# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training model
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = (correct / total) * 100
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')

    return model

def main():
    set_seed(42)
    # ['SD', 'LOSO', 'FT']

    # load train dataset
    data_type = 'SD' # for save model path

    train_dataset = MIBCI2aDataset(mode='train', data_type='SD')
    train_dataloader = DataLoader(train_dataset, batch_size=300, shuffle=True)

    # initialize model
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=8, dropoutRate=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # train model
    trained_model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=1111)

    if data_type == 'FT':
        # load finetune dataset
        finetune_dataset = MIBCI2aDataset(mode='finetune', data_type='FT')
        finetune_dataloader = DataLoader(finetune_dataset, batch_size=300, shuffle=True)

        model_path = f'sccnet_LOSO_model.pth'
        model.load_state_dict(torch.load(model_path))

        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

        # finetuned model
        finetuned_model = train_model(model, finetune_dataloader, criterion, optimizer, num_epochs=287)
    
        # save model
        torch.save(finetuned_model.state_dict(), f'sccnet_{data_type}_model.pth')
        print(f'Model saved for {data_type}')

    else:
        # save model
        torch.save(trained_model.state_dict(), f'sccnet_{data_type}_model.pth')
        print(f'Model saved for {data_type}')

if __name__ == '__main__':
    main()
