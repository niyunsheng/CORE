
import torch
from PIL import Image 
import os, json, glob
import cv2
import logging

from .base_dataset import BaseDataset

from utils import log_print

class FFpp(BaseDataset):
    def __init__(self,root,train_type="train",transform=None,num_classes=2,quality='c23'):
        super(FFpp,self).__init__(root=root, transform=transform, num_classes=num_classes)

        fake_imgs = []
        real_imgs = []
        fake_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        
        if train_type=="train":
            with open(os.path.join(root,"split/train.json")) as f:
                pairs = json.load(f)
        elif train_type=="val":
            with open(os.path.join(root,"split/val.json")) as f:
                pairs = json.load(f)
        elif train_type=="test":
            with open(os.path.join(root,"split/test.json")) as f:
                pairs = json.load(f)

        for pair in pairs:
            a,b = pair
            for fake_type in fake_types:
                fake_imgs += [[t,1] for t in glob.glob(os.path.join(root,"manipulated_sequences/{}/{}/videos".format(fake_type,quality),"{}_{}".format(a,b),"*.png"))]
                fake_imgs += [[t,1] for t in glob.glob(os.path.join(root,"manipulated_sequences/{}/{}/videos".format(fake_type,quality),"{}_{}".format(b,a),"*.png"))]
            real_imgs += [[t,0] for t in glob.glob(os.path.join(root,"original_sequences/youtube/{}/videos".format(quality),"{}".format(a),"*.png"))]
            real_imgs += [[t,0] for t in glob.glob(os.path.join(root,"original_sequences/youtube/{}/videos".format(quality),"{}".format(b),"*.png"))]

        log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

        self.imgs = fake_imgs + real_imgs
        # import random
        # self.imgs = random.sample(self.imgs, 64)
