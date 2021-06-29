from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
import math
from joblib import Parallel, delayed

def eliminate_light(fr):
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimg1, 220, 255, cv2.THRESH_BINARY)[1]
    result2 = cv2.inpaint(fr, mask2, 0.1, cv2.INPAINT_TELEA)
    return result2

fps = 30
divided = fps * 60 * 10

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

video = mmcv.VideoReader('./result_r.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

frames = [frames[i:i+divided] for i in range(0, len(frames), divided)]

cnt = 0

def process_epoch_signals(frames):
    frames_tracked = []
    frames_croped = []
    for i, frame in tqdm(enumerate(frames)):

        try:
            # Detect faces
            boxes, _ = mtcnn.detect(frame)

            # Draw faces
            frame_draw = frame.copy()
            #draw = ImageDraw.Draw(frame_draw)
            #for box in boxes:
            #    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            # Add to frame list
            my_frame = cv2.cvtColor(np.array(frame_draw.resize((640, 480), Image.BILINEAR)), cv2.COLOR_RGB2BGR)
            my_frame = eliminate_light(my_frame)
            my_frame_crop = my_frame[math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), math.ceil(boxes[0][0]):math.ceil(boxes[0][2])]
            my_frame_crop = cv2.resize(my_frame_crop, (300, 300), interpolation=cv2.INTER_AREA)
            frames_tracked.append(my_frame)
            frames_croped.append(my_frame_crop)

        except:
            frame_t = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frames_tracked.append(frame_t)
            my_frame_crop = cv2.resize(frame_t, (300, 300), interpolation=cv2.INTER_AREA)
            frames_croped.append(my_frame_crop)
    return frames_tracked, frames_croped


def process_framse(frames_tracked, video_tracked):
    for frame in frames_tracked:
        video_tracked.write(frame)

def save_video(tt, frames_tracked, frames_croped):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_tracked = cv2.VideoWriter('./video_tracked_10min_{}.mp4'.format(tt), fourcc, 30, (640, 480))
    video_tracked_crop = cv2.VideoWriter('./video_tracked_crop_10min_{}.mp4'.format(tt), fourcc, 30, (300, 300))
    process_framse(frames_tracked, video_tracked)
    process_framse(frames_croped, video_tracked_crop)
    video_tracked.release()
    video_tracked_crop.release()

def process_combine(tt, frame_t):
    frames_tracked, frames_croped = process_epoch_signals(frame_t)
    save_video(tt, frames_tracked, frames_croped)
    del frames_tracked
    del frames_croped
    print('The {} epoch has done'.format(tt))


for tt, frame_t in tqdm(enumerate(frames)):
    frames_tracked, frames_croped = process_epoch_signals(frame_t)
    save_video(tt, frames_tracked, frames_croped)
    del frames_tracked
    del frames_croped
    print('The {} epoch has done'.format(tt))

print('\nDone')


