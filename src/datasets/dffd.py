
import torch
from PIL import Image 
import os, json, glob
import cv2

from .base_dataset import BaseDataset
from utils import log_print

class DFFD(BaseDataset):
    def __init__(self,root,train_type="train",transform=None,num_classes=2):
        super(DFFD,self).__init__(root=root, transform=transform, num_classes=num_classes)

        real_dirs = ["youtube", "ffhq", "celeba_2w"]

        fake_dirs = ["stylegan_celeba", "stylegan_ffhq", "faceapp", "stargan", "pggan_v1", "pggan_v2", "Deepfakes", "FaceSwap", "Face2Face"]

        real_imgs = []
        for d in real_dirs:
            tmp_list = glob.glob(os.path.join(root, d, train_type, "*.png"))
            print(d, len(tmp_list))
            real_imgs += tmp_list
        fake_imgs = []
        for d in fake_dirs:
            tmp_list = glob.glob(os.path.join(root, d, train_type, "*.png"))
            print(d, len(tmp_list))
            fake_imgs += tmp_list
        
        log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

        self.imgs = [[p,1] for p in fake_imgs] + [[p,0] for p in real_imgs]
