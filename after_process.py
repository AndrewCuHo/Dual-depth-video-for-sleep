from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
import math

from joblib import Parallel, delayed


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

video = mmcv.VideoReader('./result_r.avi')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
print()
frames_tracked = []
frames_croped = []

for i, frame in tqdm(enumerate(frames)):
    #print('\rTracking frame: {}'.format(i + 1), end='')

    try:
        # Detect faces
        boxes, _ = mtcnn.detect(frame)

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        # Add to frame list
        my_frame = cv2.cvtColor(np.array(frame_draw.resize((640, 480), Image.BILINEAR)), cv2.COLOR_RGB2BGR)
        my_frame_crop = my_frame[math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), math.ceil(boxes[0][0]):math.ceil(boxes[0][2])]
        my_frame_crop = cv2.resize(my_frame_crop, (300, 300), interpolation=cv2.INTER_AREA)
        frames_tracked.append(my_frame)
        frames_croped.append(my_frame_crop)

    except:
        frame_t = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frames_tracked.append(frame_t)
        my_frame_crop = cv2.resize(frame_t, (300, 300), interpolation=cv2.INTER_AREA)
        frames_croped.append(my_frame_crop)

print('\nDone')

fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
video_tracked = cv2.VideoWriter('./video_tracked.avi', fourcc, 30, (640, 480))
video_tracked_crop = cv2.VideoWriter('./video_tracked_crop.avi', fourcc, 30, (300, 300))

def process_framse(frames_tracked, video_tracked):
    for frame in frames_tracked:
        video_tracked.write(frame)

process_framse(frames_tracked, video_tracked)
process_framse(frames_croped, video_tracked_crop)
video_tracked.release()
video_tracked_crop.release()

