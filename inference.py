from torchvision.utils import draw_segmentation_masks
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import ImageReadMode
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

#from model import model


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def main():

    test_set_dir = './dataset'

    img_path = os.path.join(test_set_dir, "plant034_rgb.png")

    test_img = read_image(path=img_path,mode=ImageReadMode.RGB)


    test_set_list = [test_img]


    grid = make_grid(test_set_list)
    show(grid)

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Step 1: Initialize model with the best available weights
    model = lraspp_mobilenet_v3_large(num_classes=2)
    model_path = './checkpoint.pth'
    checkpoint = torch.load(model_path)
    #print(checkpoint)
    #print(checkpoint[model])
    weights = checkpoint['model']
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
 
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(test_img).unsqueeze(0)

    batch = batch.to(device)
    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    #class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    #classes = [background, plant]
    mask = normalized_masks[0, 0]
    show(mask)
    bool_mask = mask.to(dtype=torch.bool)
    print('bool mask shape',bool_mask.shape)
    print('img shape',test_img.size())

    resized_img = T.Resize(size=[bool_mask.size(dim=0),bool_mask.size(dim=1)])(test_img)
    print('resized img shape',resized_img.size())

    resized_test_images = [resized_img]
    masks = [bool_mask]
    masked_imgs = [
        draw_segmentation_masks(img, masks=mask, alpha=1)
        for img, mask in zip(resized_test_images, masks)
    ]
    show(masked_imgs)                                

if __name__ == "__main__":
    #args = get_args_parser().parse_args()
    main()
