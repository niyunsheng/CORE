
import torch
from PIL import Image 
import os, json, glob
import cv2

from .base_dataset import BaseDataset
from utils import log_print

class DFDCP(BaseDataset):
    def __init__(self,root,train_type="train",transform=None,num_classes=2):
        super(DFDCP,self).__init__(root=root, transform=transform, num_classes=num_classes)

        metadata_path = os.path.join(root, "dataset.json")
        with open(metadata_path,'r')as fp:
            metadata = json.load(fp)
        assert train_type in ["train","test"]
        videos = [k for k in metadata.keys() if metadata[k]["set"]==train_type]
        
        real_imgs = []
        fake_imgs = []
        for video in videos:
            if metadata[video]["label"] == "real":
                real_imgs += glob.glob(os.path.join(root, video[:-4], "*.png"))
            else:
                fake_imgs += glob.glob(os.path.join(root, video[:-4], "*.png"))

        log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

        self.imgs = [[p,1] for p in fake_imgs] + [[p,0] for p in real_imgs]
