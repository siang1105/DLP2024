import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet
from Dataloader import MIBCI2aDataset

def evaluate_model(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100
    return avg_loss, accuracy

def main():
    # ['SD', 'LOSO', 'FT']
    # load test dataser
    data_type = 'SD' # for changing model

    test_dataset = MIBCI2aDataset(mode='test', data_type='SD')
    test_dataloader = DataLoader(test_dataset, batch_size=300, shuffle=False)

    if len(test_dataset) == 0:
        print('No test data found.')
        return

    # initialize model
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=1, dropoutRate=0.5)
    model_path = f'sccnet_{data_type}_model.pth'
    
    # load model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded for {data_type}')
    else:
        print(f'Model file not found for {data_type}, skipping...')

    criterion = nn.CrossEntropyLoss()

    # evaluate
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion)
    print(f'Test Loss for {data_type}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
