import torch
from torch.utils import data
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random

class MRI_Dataset(data.Dataset):
    def __init__ (self, file_list, transforms = None):
        self.file_list = file_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.file_list)
    
    '''
    def __getitem__(self, index):
        image_path = self.file_list[index][0]
        mask_path = self.file_list[index][1]

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        img_numpy = np.transpose(np.array(img), (2, 0, 1))
        mask_numpy = np.resize(np.array(mask)/255, (1, 256, 256))

        x = torch.from_numpy(img_numpy).type(torch.float32)
        y = torch.from_numpy(mask_numpy).type(torch.float32)

        if self.transforms is not None:
            x = self.transforms(x)
            y = self.transforms(y)

        return x, y
    '''

    def __getitem__(self, idx):

        img_path = self.file_list[idx][0]
        img = Image.open(img_path).convert('RGB')

        mask_path = self.file_list[idx][1]
        mask_img = Image.open(mask_path).convert('L')

        resize = transforms.Resize(size=(256, 256))
        img = resize(img)
        mask_img = resize(mask_img)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask_img  = TF.hflip(mask_img)

        if random.random() > 0.5:
            img = TF.vflip(img)
            mask_img  = TF.vflip(mask_img)

        '''
        if random.random() > 0.5:
            img = TF.affine(img, angle=15, translate=(0.1, 0.1), scale=(0.8, 0.8), shear = 0)
            mask_img = TF.affine(mask_img, angle=15, translate=(0.1, 0.1), scale=(0.8, 0.8), shear = 0)
        '''
        img = TF.to_tensor(img)
        mask_img = TF.to_tensor(mask_img)

        return img, mask_img