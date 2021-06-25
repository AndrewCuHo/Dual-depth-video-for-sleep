import cv2
import time
import datetime
from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw

def capture(mode='video'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

    camera_r = cv2.VideoCapture(0)
    camera_r.set(cv2.CAP_PROP_FRAME_WIDTH, 640)#1280
    camera_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#480

    fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')  # 视频编解码器
    fps = 20  # camera.get(cv2.CAP_PROP_FPS)  # 帧数
    out_r = cv2.VideoWriter('./Savevideo/result_r.avi', fourcc, fps, (640, 480))  # 写入视频

    # ret, frame = camera.read()
    # ret_r, frame_r = camera_r.read()

    while camera_r.isOpened():
        ret_r, frame_r = camera_r.read()
        if ret_r == True:
            # setting time
            date_now = str(datetime.datetime.now())
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame_r = cv2.putText(frame_r, date_now, (5, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            out_r.write(frame_r)  # 写入帧
            cv2.imshow("cam_r", frame_r)

            #video = mmcv.VideoReader('./Savevideo/result_r.avi')
            frame = Image.fromarray(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
            # Detect faces
            boxes = mtcnn.detect(frame)

            # Draw faces
            frame_draw = frame.copy()
            frame_crop = frame.copy()
            frame_crop = frame_crop[points[0]:points[1], points[2]:points[3], :]

            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

            # Add to frame list
            frames_tracked = frame_draw.resize((640, 480), Image.BILINEAR)
            frames_focused = frame_crop.resize((640, 480), Image.BILINEAR)

            cv2.imshow("cam_track", cv2.cvtColor(np.array(frames_tracked), cv2.COLOR_RGB2BGR))
            cv2.imshow("cam_focused", cv2.cvtColor(np.array(frames_focused), cv2.COLOR_RGB2BGR))


            if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
                break
        else:
            break

    camera_r.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()


