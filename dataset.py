import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import torch.utils.data as data
import torch
from PIL import Image

class PlantNetDataset:
    def __init__(self, img_dir, transforms):
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(img_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(img_dir, "masks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.img_dir, "masks", self.masks[idx])
        #img = read_image(img_path,mode=ImageReadMode.RGB)
        img = Image.open(img_path).convert('RGB')
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #target = read_image(mask_path)
        target = Image.open(mask_path)

        #target = np.array(mask)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
