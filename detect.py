import cv2
import time
import datetime
from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
import math

def capture(mode='video', width=1280, height=480, if_dua=True):
    if not if_dua:
        width = width//2
    device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)
    if if_dua:
        cv2.namedWindow("left")
        cv2.namedWindow("right")

    camera_r = cv2.VideoCapture(0)
    camera_r.set(cv2.CAP_PROP_FRAME_WIDTH, width)#1280
    camera_r.set(cv2.CAP_PROP_FRAME_HEIGHT, height)#480

    fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')  # 视频编解码器
    fps = 20  # camera.get(cv2.CAP_PROP_FPS)  # 帧数
    if if_dua:
        out_left = cv2.VideoWriter('./Savevideo/result_left.avi', fourcc, fps, (width//2, height))  # 写入视频
        out_right = cv2.VideoWriter('./Savevideo/result_right.avi', fourcc, fps, (width//2, height))  # 写入视频
    else:
        out_r = cv2.VideoWriter('./Savevideo/result_r.avi', fourcc, fps, (width, height))  # 写入视频

    # ret, frame = camera.read()
    # ret_r, frame_r = camera_r.read()

    while camera_r.isOpened():
        ret_r, frame_r = camera_r.read()
        if ret_r == True:
            # setting time
            date_now = str(datetime.datetime.now())
            font = cv2.FONT_HERSHEY_SIMPLEX

            if if_dua:
                left_frame = frame_r[0:height, 0:width//2]# 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
                right_frame = frame_r[0:height, width//2:width]


                left_frame = cv2.putText(left_frame, date_now, (5, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)


                # video = mmcv.VideoReader('./Savevideo/result_r.avi')
                frame_l = Image.fromarray(cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB))
                frame_r = Image.fromarray(cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB))
                # Detect faces
                boxes_l, _ = mtcnn.detect(frame_l)
                boxes_r, _ = mtcnn.detect(frame_r)

                # Draw faces
                frame_draw_l = frame_l.copy()
                right_frame = right_frame[math.ceil(boxes_r[0][1]):math.ceil(boxes_r[0][3]), math.ceil(boxes_r[0][0]):math.ceil(boxes_r[0][2])]

                draw_l = ImageDraw.Draw(frame_draw_l)

                for box in boxes_l:
                    draw_l.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                # Add to frame list
                frames_tracked_l = frame_draw_l.resize((width//2, height), Image.BILINEAR)

                left_frame = cv2.cvtColor(np.array(frames_tracked_l), cv2.COLOR_RGB2BGR)
                right_frame = cv2.resize(right_frame, (width//2, height), interpolation = cv2.INTER_AREA)
                right_frame = cv2.putText(right_frame, date_now, (5, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

                out_left.write(left_frame)  # 写入帧
                out_right.write(right_frame)  # 写入帧

                cv2.imshow("cam_track_l", left_frame)
                cv2.imshow("cam_track_r", right_frame)

            else:
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
                #frame_crop = frame_crop[points[0]:points[1], points[2]:points[3], :]

                draw = ImageDraw.Draw(frame_draw)
                for box in boxes:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                # Add to frame list
                frames_tracked = frame_draw.resize((width, height), Image.BILINEAR)
                #frames_focused = frame_crop.resize((width, height), Image.BILINEAR)

                cv2.imshow("cam_track", cv2.cvtColor(np.array(frames_tracked), cv2.COLOR_RGB2BGR))
                #cv2.imshow("cam_focused", cv2.cvtColor(np.array(frames_focused), cv2.COLOR_RGB2BGR))


            if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
                break
        else:
            break

    camera_r.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()
