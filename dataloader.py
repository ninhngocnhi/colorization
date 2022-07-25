import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import random

def gen_test_img(imgPath):
    gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    for i in range(0, len(imgPath)):
        img = cv2.imread(imgPath[i])
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray= img[:, :, 2]
        gray = gray_transform(gray)
        gray = torch.unsqueeze(gray, 0)
        if i == 0:
            input = gray
        else:
            input = torch.cat((input, gray), 0)

    return input


class CustomDataset(data.Dataset):
    def __init__(self, imgPath):
        super(CustomDataset, self).__init__()
        self.imgPath = imgPath
    def transform(self, gray, image):
        gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if random.random() > 0.5:
            gray = TF.hflip(gray)
            image = TF.hflip(image)
        if random.random() > 0.5:
            tmp = random.randint(20, 50)
            gray = TF.rotate(gray, tmp)
            image = TF.rotate(image, tmp)
        if random.random() > 0.5:
            gray = TF.vflip(gray)
            image = TF.vflip(image)
        if random.random() > 0.5:
            tmp = random.randint(20, 50)
            affine = transforms.Compose([
                transforms.RandomAffine(tmp)
            ])
            gray = affine(gray)
            image = affine(image)

        gray = gray_transform(gray)
        image = normal_transform(image)
        return gray, image

    def __getitem__(self, idx):
        img = cv2.imread(self.imgPath[idx])
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray= img[:, :, 2]
        gray = Image.fromarray(np.uint8(gray), 'L')
        img = Image.fromarray(img)
        gray, img = self.transform(gray, img)
        
        return gray, img
    
    def __len__(self):
        return len(self.imgPath)
