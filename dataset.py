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
        self.instance_masks = list(sorted(os.listdir(os.path.join(img_dir, "instance-masks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.img_dir, "masks", self.masks[idx])
        instance_mask_path = os.path.join(self.img_dir, "instance-masks", self.instance_masks[idx])
        #img = read_image(img_path,mode=ImageReadMode.RGB)
        img = Image.open(img_path).convert('RGB')
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #target = read_image(mask_path)
        if self.masks is not None:
            target = Image.open(mask_path)
        else:
            #create masks here from instance masks
            target = Image.open(instance_mask_path)
            #convert to bool mask

            pass
        #target = np.array(mask)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
