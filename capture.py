import cv2
import time
import datetime

mode = 'video'
AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2  # 自动拍照间隔

cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0)
camera_r = cv2.VideoCapture(2)

# 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率1280x480；分割为两个640x480、640x480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)#1280
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#480

camera_r.set(cv2.CAP_PROP_FRAME_WIDTH, 640)#1280
camera_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#480

counter = 0
utc = time.time()
folder = "/home/pi/Desktop/sleep/SaveImage/"  # 拍照文件目录

if mode == 'image':
    def shot(pos, frame):
        global counter
        path = folder + pos + "_" + str(counter) + ".jpg"

        cv2.imwrite(path, frame)
        print("snapshot saved into: " + path)


    while True:
        ret, frame = camera.read()
        ret_r, frame_r = camera_r.read()
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        left_frame = frame[0:480, 0:640]
        right_frame = frame[0:480, 640:1280]

        cv2.imshow("left", left_frame)
        cv2.imshow("right", right_frame)
        cv2.imshow("cam_r", frame_r)

        now = time.time()
        if AUTO and now - utc >= INTERVAL:
            shot("left", left_frame)
            shot("right", right_frame)
            shot("cam_r", frame_r)
            counter += 1
            utc = now

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot("left", left_frame)
            shot("right", right_frame)
            shot("cam_r", frame_r)
            counter += 1
    camera.release()
    camera_r.release()
    cv2.destroyWindow("left")
    cv2.destroyWindow("right")
    cv2.destroyWindow("cam_r")
else:
    fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')  # 视频编解码器
    fps = 20 #camera.get(cv2.CAP_PROP_FPS)  # 帧数
    #fps = 30
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    out_left = cv2.VideoWriter('/home/pi/Desktop/sleep/Savevideo/result_left.avi', fourcc, fps, (640, 480))  # 写入视频
    out_right = cv2.VideoWriter('/home/pi/Desktop/sleep/Savevideo/result_right.avi', fourcc, fps, (640, 480))  # 写入视频
    out_r = cv2.VideoWriter('/home/pi/Desktop/sleep/Savevideo/result_r.avi', fourcc, fps, (640, 480))  # 写入视频

    #ret, frame = camera.read()
    #ret_r, frame_r = camera_r.read()

    while camera.isOpened():
        ret, frame = camera.read()
        ret_r, frame_r = camera_r.read()
        if ret == True & ret_r == True:
            # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
            left_frame = frame[0:480, 0:640]
            right_frame = frame[0:480, 640:1280]
            # setting time
            date_now = str(datetime.datetime.now())
            font = cv2.FONT_HERSHEY_SIMPLEX

            left_frame = cv2.putText(left_frame, date_now, (5, 10), font, 0.3, (0, 255,255), 1, cv2.LINE_AA)
            right_frame = cv2.putText(right_frame, date_now, (5, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
            frame_r = cv2.putText(frame_r, date_now, (5, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            out_left.write(left_frame)  # 写入帧
            out_right.write(right_frame)  # 写入帧
            out_r.write(frame_r)  # 写入帧
            cv2.imshow("left", left_frame)
            cv2.imshow("right", right_frame)
            cv2.imshow("cam_r", frame_r)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
                break
        else:
            break

    camera.release()
    camera_r.release()
    cv2.destroyWindow("left")
    cv2.destroyWindow("right")
    cv2.destroyWindow("cam_r")
