
import torch
from PIL import Image 
import os, json, glob
import cv2
import pandas as pd

from .base_dataset import BaseDataset
from utils import log_print

class CelebDF(BaseDataset):
    def __init__(self,root,train_type="train",transform=None,num_classes=2):
        super(CelebDF,self).__init__(root=root, transform=transform, num_classes=num_classes)

        csv_path = os.path.join(root, "List_of_testing_videos.txt")
        videos = [t.replace(".mp4","") for t in pd.read_csv(csv_path, sep=" ", header=None).values[:,1].tolist()]
        if train_type == "train":
            test_videos = videos
            all_videos = list(filter(lambda x: os.path.isdir(x), glob.glob(os.path.join(root, "*/*"))))
            all_videos = [t.replace(root+"/", "") for t in all_videos]
            videos = list(set(all_videos)-set(test_videos))
        
        real_imgs = []
        fake_imgs = []
        for video in videos:
            if "real" in video:
                real_imgs += glob.glob(os.path.join(root, video, "*.png"))
            else:
                fake_imgs += glob.glob(os.path.join(root, video, "*.png"))
        
        log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

        fake_imgs = [[p,1] for p in fake_imgs]
        real_imgs = [[p,0] for p in real_imgs]
        self.imgs = fake_imgs + real_imgs
