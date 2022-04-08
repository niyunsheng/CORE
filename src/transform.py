from PIL import Image
import numpy as np

import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import torch
import random

class TwoTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return [v1, v2]

class OneOfTrans:
    """random select one of from the input transform list"""

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return self.base_transforms[random.randint(0,len(self.base_transforms)-1)](x)

class ALBU_AUG:
    def __init__(self, base_transform):
        self.transform = base_transform
    
    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        return self.transform(image=x)['image']

def get_augs(name="base", norm="imagenet", size=299):
    IMG_SIZE = size
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "0.5":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    if name == "None":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RE":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RandCrop":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RaAug":
        return OneOfTrans([
            transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])
        ])
    elif name == "DFDC_selim":
        # dfdc 第一名数据增强方案
        return ALBU_AUG(A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.HorizontalFlip(),
            A.OneOf([
                A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_CUBIC),
                A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_AREA),
                A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            ], p=1.0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Normalize(mean=tuple(mean),std=tuple(std)),
            ToTensorV2()
        ]))
    else:
        raise NotImplementedError