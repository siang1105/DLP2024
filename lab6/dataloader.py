import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np

class iclevrDataset(Dataset):
    def __init__(self, root=None, mode="train", transform=None):
        super().__init__()
        assert mode in ["train", "test", "new_test"], "Invalid mode specified!"

        self.root = root
        self.mode = mode
        self.transform = transform

        self.json_data = self._load_json(f'{self.mode}.json')
        self.objects_dict = self._load_json('objects.json')

        if self.mode == "train":
            self.img_paths, self.labels = list(self.json_data.keys()), list(self.json_data.values())
        elif self.mode == "test" or self.mode == "new_test":
            self.labels = self.json_data

        self.labels_one_hot = torch.zeros(len(self.labels), len(self.objects_dict)) #[樣本數, 類別數]

        for i, label_list in enumerate(self.labels):
            # For each label in each label list, get the corresponding index
            label_indices = []
            for label in label_list:
                index = self.objects_dict[label]
                label_indices.append(index)

            # Set the one-hot position corresponding to these indexes to 1
            self.labels_one_hot[i][label_indices] = 1

    def _load_json(self, filepath):
        """Helper function to load JSON files."""
        with open(filepath, 'r') as json_file:
            return json.load(json_file)
            
    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = os.path.join(self.root, self.img_paths[index])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            label_one_hot = self.labels_one_hot[index]
            return img, label_one_hot
        
        elif self.mode == "test" or self.mode == "new_test":
            label_one_hot = self.labels_one_hot[index]
            return label_one_hot

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = iclevrDataset(root='iclevr', mode='train', transform=transform)
    print(len(dataset))