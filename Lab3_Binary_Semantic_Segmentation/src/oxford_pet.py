import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import torchvision.transforms as transforms 
import albumentations as alb #高效的圖像增強庫
from albumentations.pytorch import ToTensorV2 #albumentations 中的模塊，用於將圖像轉換為 PyTorch 張量

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        # sample = dict(image=image, mask=mask, trimap=trimap)
        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)
            sample["mask"] = sample["mask"].unsqueeze(0)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0] #表示選取索引不被 10 整除的文件，即 90% 的數據
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0] #表示選取索引被 10 整除的文件，即 10% 的數據
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    #定義了訓練數據的圖像增強和預處理步驟。這些增強步驟有助於增加數據的多樣性，提高模型的泛化能力。
    train_transform = alb.Compose([
        alb.Resize(256, 256), #將圖像調整為 256x256 像素
        alb.HorizontalFlip(), #隨機水平翻轉圖像
        alb.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)), #隨機裁剪圖像並調整為 256x256 像素，裁剪比例在 80% 到 100% 之間
        alb.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, p=0.5), #隨機平移、縮放和旋轉圖像
        alb.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, p=0.5), #隨機調整圖像的亮度、對比度、飽和度和色調
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #將圖像的每個通道進行標準化，使其具有零均值和單位標準差
        ToTensorV2(), #將圖像轉換為 PyTorch 張量
    ])

    #定義了驗證和測試數據的圖像預處理步驟，這些步驟不包含數據增強，僅進行必要的標準化和格式轉換
    transform = alb.Compose([
        alb.Resize(256, 256),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    if mode == "train":
        dataset = OxfordPetDataset(root=data_path, mode="train", transform=train_transform)
    elif mode == "valid":
        dataset = OxfordPetDataset(root=data_path, mode="valid", transform=transform)
    elif mode == "test":
        dataset = OxfordPetDataset(root=data_path, mode="test", transform=transform)

    return dataset