import cv2
import time
import datetime
from facenet_pytorch import MTCNN, extract_face
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
import math
from precrocess import process_frame, align_faces, aligned_face, eliminate_light
from PyQt5 import QtWidgets,QtCore,QtGui
import pyqtgraph as pg
import sys
import traceback
import matplotlib.pyplot as plt

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG监控 - chaunhao")
        self.main_widget = QtWidgets.QWidget()  # 创建一个主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建一个网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置主部件的布局为网格
        self.setCentralWidget(self.main_widget)  # 设置窗口默认部件

        self.plot_widget = QtWidgets.QWidget()  # 实例化一个widget部件作为K线图部件
        self.plot_layout = QtWidgets.QGridLayout()  # 实例化一个网格布局层
        self.plot_widget.setLayout(self.plot_layout)  # 设置K线图部件的布局层
        self.plot_plt = pg.PlotWidget()  # 实例化一个绘图部件
        self.plot_plt.showGrid(x=True, y=True)  # 显示图形网格
        self.plot_layout.addWidget(self.plot_plt)  # 添加绘图部件到K线图部件的网格布局层
        # 将上述部件添加到布局层中
        self.main_layout.addWidget(self.plot_widget, 1, 0, 3, 3)

        self.setCentralWidget(self.main_widget)
        self.plot_plt.setYRange(max=100, min=0)
        self.data_list = []
        self.timer_start()

    # 启动定时器 时间间隔秒
    def timer_start(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.get_cpu_info)
        self.timer.start(1000)

    # 获取CPU使用率
    def get_cpu_info(self):
        try:
            cpu = "%0.2f" % psutil.cpu_percent(interval=1)
            self.data_list.append(float(cpu))
            print(float(cpu))
            self.plot_plt.plot().setData(self.data_list, pen='g')
        except Exception as e:
            print(traceback.print_exc())


def capture(mode='video', width=1280, height=480, if_dua=True):
    if not if_dua:
        width = width//2
    align_width = 112
    align_height = 112

    device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

    camera_r = cv2.VideoCapture(0)
    camera_r.set(cv2.CAP_PROP_FRAME_WIDTH, width)#1280
    camera_r.set(cv2.CAP_PROP_FRAME_HEIGHT, height)#480

    fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')  # 视频编解码器
    fps = 30  # camera.get(cv2.CAP_PROP_FPS)  # 帧数
    if if_dua:
        out_left = cv2.VideoWriter('./Savevideo/result_left.avi', fourcc, fps, (width//2, height))  # 写入视频
        out_right = cv2.VideoWriter('./Savevideo/result_right.avi', fourcc, fps, (align_width, align_height))  # 写入视频
    else:
        out_r = cv2.VideoWriter('./Savevideo/result_r.avi', fourcc, fps, (width, height))  # 写入视频

    # ret, frame = camera.read()
    # ret_r, frame_r = camera_r.read()
    right_frames = []
    date_1 = int(time.time())

    while camera_r.isOpened():
        ret_r, frame_r = camera_r.read()
        if ret_r == True:
            # setting time
            date_now = str(datetime.datetime.now())
            date_2 = int(time.time())
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
                right_frame = cv2.resize(right_frame, (align_width, align_height), interpolation=cv2.INTER_AREA)
                right_frame = eliminate_light(right_frame)
                #right_frame = cv2.putText(right_frame, date_now, (5, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
                #right_frame = cv2.fastNlMeansDenoisingColored(right_frame, None, 3, 3, 3, 21)
                #similar_trans_matrix = align_faces(landmark)
                #aligned_face = cv2.warpAffine(right_frame.copy(), similar_trans_matrix, (width//2, height))
                #boxes_r, _ = mtcnn.detect(aligned_faced, landmarks=False)
                #right_frame = aligned_faced[math.ceil(boxes_r[0][1]):math.ceil(boxes_r[0][3]), math.ceil(boxes_r[0][0]):math.ceil(boxes_r[0][2])]

                out_left.write(left_frame)  # 写入帧
                out_right.write(right_frame)  # 写入帧
                right_frames.append(right_frame)

                if date_2 - date_1 > 30:
                    HR = process_frame(right_frames, fps)
                    right_frames = []
                    date_1 = date_2
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(HR)
                    plt.ion()  # 将画图模式改为交互模式

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
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    capture()
