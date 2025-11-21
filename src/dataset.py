from torch.utils.data import Dataset
from PIL import Image
import os

class PlantDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform

        self.img_path = []
        self.labels = []

        class_folders = sorted(os.listdir(root))
        self.class_names = class_folders

        for idx, folder in enumerate(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            for img in os.listdir(folder_path):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.img_path.append(os.path.join(folder_path, img))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert("RGB")
        label = self.labels[item]

        if self.transform:
            img = self.transform(img)
        return img, label