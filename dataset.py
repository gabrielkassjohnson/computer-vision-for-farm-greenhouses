from genericpath import exists
import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

class PlantNetDataset:
    def __init__(self, img_dir, transforms):
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(img_dir, "images"))))
        #self.masks = list(sorted(os.listdir(os.path.join(img_dir, "masks"))))
        self.instance_masks = list(sorted(os.listdir(os.path.join(img_dir, "instance-masks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "images", self.imgs[idx])
        img = Image.open(img_path).convert('RGB')


        #create masks here from instance masks
        instance_mask_path = os.path.join(self.img_dir, "instance-masks", self.instance_masks[idx])

        instance_mask = Image.open(instance_mask_path)
        #convert to mask with only 0 and 1 as values
        array = np.array(instance_mask)
        array[array > 0] = 1
        target = Image.fromarray(array)
        #print('converted', target,'target')
        #target.save(os.path.join('../masks',file.name))
        #print(file.name)


        #target = np.array(mask)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
