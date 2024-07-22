import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        features = []
        for file in os.listdir(filePath):
            feature = np.load(os.path.join(filePath, file))
            features.append(feature)
        return np.concatenate(features, axis=0)

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        labels = []
        for file in os.listdir(filePath):
            label = np.load(os.path.join(filePath, file))
            labels.append(label)
        return np.concatenate(labels, axis=0)

    def __init__(self, mode, subject_id=None, data_type='SD'):
        assert mode in ['train', 'test', 'finetune']
        self.features = np.array([])
        self.labels = np.array([])

        if data_type == 'SD':
            if mode == 'train':
                feature_path = f'./dataset/SD_train/features/'
                label_path = f'./dataset/SD_train/labels/'
            if mode == 'test':
                feature_path = f'./dataset/SD_test/features/'
                label_path = f'./dataset/SD_test/labels/'

        elif data_type == 'LOSO':
            if mode == 'train':
                feature_path = './dataset/LOSO_train/features/'
                label_path = './dataset/LOSO_train/labels/'
            elif mode == 'test':
                feature_path = './dataset/LOSO_test/features/'
                label_path = './dataset/LOSO_test/labels/'

        elif data_type == 'FT':
            if mode == 'finetune':
                feature_path = './dataset/FT/features/'
                label_path = './dataset/FT/labels/'
            elif mode == 'test':
                feature_path = './dataset/LOSO_test/features/'
                label_path = './dataset/LOSO_test/labels/'

        self.features = self._getFeatures(feature_path)
        self.labels = self._getLabels(label_path)

        if self.features.size > 0:
            self.features = self.features[:, np.newaxis, :, :]  # Add a dimension so that the shape of the feature is (num_samples, 1, 22, 438)

    def __len__(self):
        # implement the len method
        return len(self.labels)

    def __getitem__(self, idx):
        # implement the getitem method
        feature = self.features[idx]
        label = self.labels[idx]
        # print(f"Sample {idx} shape: {feature.shape}")
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)