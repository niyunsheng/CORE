'''
preprocessing FF dataset, using mask to select the truely forgery faces
1. sample frames for videos while ensuring fps>=3
2. save face crops of sampled frames

'''
import argparse, os
import cv2
import json
import warnings
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
import torch

warnings.filterwarnings("ignore")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

CROP_NAME_PATTERN = '{:06d}.png'
EXPAND_RATIO = 0.15
BATCH_SIZE = 32

def get_detector():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN(
        margin=0,
        thresholds=[0.60, 0.60, 0.60],
        device=device,
        select_largest=True,  # the face with biggest size is ranked first
        keep_all=True)
    return face_detector

def adjust_bbox_ratio(bbox, avg_ratio, enlarge=False):
    xmin, ymin, xmax, ymax = bbox
    x_len = xmax - xmin
    y_len = ymax - ymin
    ratio = x_len / y_len
    if ratio == avg_ratio:
        return bbox
    elif ratio > avg_ratio:
        if enlarge:
            r_y_len = x_len / avg_ratio
            ymin -= (r_y_len - y_len) / 2
            ymax = ymin + r_y_len
        else:
            r_x_len = (y_len * avg_ratio)
            xmin += (x_len - r_x_len) / 2
            xmax = xmin + r_x_len
    else:
        if enlarge:
            r_x_len = y_len * avg_ratio
            xmin -= (r_x_len - x_len) / 2
            xmax = xmin + r_x_len
        else:
            r_y_len = (x_len / avg_ratio)
            ymin += (y_len - r_y_len) / 2
            ymax = ymin + r_y_len
    rect_ratio = (xmax - xmin) / (ymax - ymin)
    # if abs(rect_ratio - avg_ratio) > 1e-2:
    #     print(rect_ratio)
    assert abs(rect_ratio - avg_ratio) < 0.1
    return [xmin, ymin, xmax, ymax]

def is_fake_by_mask(mask_frame, boxes):
    xmin, ymin, xmax, ymax = boxes
    h, w, _ = mask_frame.shape
    ymin = max(ymin, 0)
    ymax = min(ymax, h)
    xmin = max(xmin, 0)
    xmax = min(xmax, w)
    # print(np.mean(mask_frame[ymin:ymax, xmin:xmax]))
    return np.mean(mask_frame[ymin:ymax, xmin:xmax])>10
    

def process_video(face_detector, video_p, crop_dir, args):
    if args.fake_type == "real":
        maks_p = video_p.replace("original_sequences/youtube/{}".format(args.quality), "manipulated_sequences/Deepfakes/masks")
        maks_p = glob(maks_p[:-4]+"*")[0]
    else:
        maks_p = video_p.replace(args.quality, "masks")
    print(video_p, maks_p)

    capture = cv2.VideoCapture(video_p)
    mask_capture = cv2.VideoCapture(maks_p)

    if not int(capture.get(7)) == int(mask_capture.get(7)):
        print(video_p, maks_p)

    video_fps = int(capture.get(cv2.CAP_PROP_FPS))
    space = video_fps // 3
    frames_num = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                     space * args.max_num)
    # print("start:", frames_num)
    # sample frames
    batch_indices = []
    batch_frames = []
    batch_mask_frames = []
    idx = 0
    while idx < frames_num:
        capture.grab()
        mask_capture.grab()
        if idx % space == 0:
            # read one frame
            success1, frame = capture.retrieve()
            success2, mask = mask_capture.retrieve()
            assert success1, "read error: {}".format(capture)
            assert success2, "read error: {}".format(mask_capture)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            batch_indices.append(idx)
            batch_frames.append(frame)
            batch_mask_frames.append(mask)

        if len(batch_frames) == BATCH_SIZE or (idx == frames_num - 1
                                               and len(batch_frames) > 0):
            # face detection
            boxes, probs, points = face_detector.detect(batch_frames,
                                                        landmarks=True)
            # save detection results
            n_frame = len(batch_frames)
            for i in range(n_frame):
                frame = cv2.cvtColor(np.asarray(batch_frames[i]),
                                     cv2.COLOR_RGB2BGR)
                frame_idx = batch_indices[i]
                frame_boxes = boxes[i]
                frame_probs = probs[i]
                frame_points = points[i]
                if frame_boxes is None:
                    continue  # no face is detected
                n_face = frame_boxes.shape[0]
                # print(n_face)
                for j in range(n_face):
                    ori_boxes = frame_boxes[j]
                    ori_points = frame_points[j]
                    square_box = adjust_bbox_ratio(ori_boxes, 1., enlarge=True)
                    square_box = [int(x) for x in square_box]
                    xmin, ymin, xmax, ymax = square_box
                    if is_fake_by_mask(batch_mask_frames[i], square_box):
                        # save cropping
                        h, w, _ = frame.shape
                        # ensure square
                        x_w = xmax - xmin
                        ymax = ymin + x_w
                        pad = int(x_w * EXPAND_RATIO)
                        ymin = max(ymin - pad, 0)
                        ymax = min(ymax + pad, h)
                        xmin = max(xmin - pad, 0)
                        xmax = min(xmax + pad, w)
                        x_w = xmax - xmin
                        y_w = ymax - ymin
                        if x_w < y_w:
                            ymin = ymin + (y_w - x_w)
                        elif x_w > y_w:
                            xmin = xmin + (x_w - y_w)
                        assert ymax - ymin == xmax - xmin
                        crop = frame[ymin:ymax, xmin:xmax]
                        crop_p = osp.join(crop_dir,
                                        CROP_NAME_PATTERN.format(frame_idx))
                        cv2.imwrite(crop_p, crop)
                        # print(frame_idx)
                        break
            # crop processing done
            batch_indices = []
            batch_frames = []
            batch_mask_frames = []

        # read next frame
        idx += 1

def delete_imgs(save_dir, max_num):
    for img in os.listdir(save_dir)[max_num:]:
        os.remove(os.path.join(save_dir, img))

def main(args):
    with open(os.path.join(args.src_root,"split/test.json")) as f:
        pairs = json.load(f)
    with open(os.path.join(args.src_root,"split/val.json")) as f:
        pairs += json.load(f)
    with open(os.path.join(args.src_root,"split/train.json")) as f:
        pairs += json.load(f)
    video_ps = []
    for a,b in pairs:
        if args.fake_type == "real":
            video_ps.append(osp.join(args.src_root, 'original_sequences/youtube/{}/videos/{}.mp4'.format(args.quality,a)))
            video_ps.append(osp.join(args.src_root, 'original_sequences/youtube/{}/videos/{}.mp4'.format(args.quality,b)))
        else:
            video_ps.append(osp.join(args.src_root, 'manipulated_sequences/{}/{}/videos/{}_{}.mp4'.format(args.fake_type, args.quality,a,b)))
            video_ps.append(osp.join(args.src_root, 'manipulated_sequences/{}/{}/videos/{}_{}.mp4'.format(args.fake_type, args.quality,b,a)))

    video_ps.sort()
    video_ps = video_ps[args.start:args.end]
    print("fake_type: ", args.fake_type)
    face_detector = get_detector()
    for video in tqdm(video_ps):
        save_dir = video.replace(args.src_root,args.dst_root).replace(".mp4","")
        os.makedirs(save_dir, exist_ok=True)
        if len(os.listdir(save_dir)) < args.max_num/2:
            process_video(face_detector, video, save_dir, args)
            print(len(os.listdir(save_dir)),save_dir)
        elif len(os.listdir(save_dir)) > args.max_num:
            delete_imgs(save_dir, args.max_num)
            print(len(os.listdir(save_dir)),save_dir)

    print('[Done]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--quality', type=str, default='c23')
    arg('--gpus', type=str, default='0')
    arg('--fake-type', type=str, default='Deepfakes',choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "real"])
    arg('--src-root', type=str, default='data/FF')
    arg('--dst-root', type=str, default='data/FF_nys_prep_with_mask')
    arg('--batch-szie', type=int, default=32)
    arg('--max-num', type=int, default=30)
    arg('--start', type=int, default=0)
    arg('--end', type=int, default=1000)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)

