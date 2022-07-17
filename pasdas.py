import cv2
import time
import os
import numpy as np
import mediapipe as mp
from requests import post
import moviepy.editor as me
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
# khoi tạo thư viện mediapipe

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "BODYSWING"
no_of_frames = 600

# vc = cv2.VideoCapture("108.mp4")
vc = cv2.VideoCapture(0)

net = cv2.dnn.readNet("D:/python/test/yolov4-tiny.weights", "D:/python/test/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
i=0
scale_percent = 100
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    width = int(frame.shape[1]* scale_percent/100)
    height = int(frame.shape[0]*scale_percent/100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    if not grabbed:
        exit()
    start = time.time()
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #making image writeable to false improves prediction
    image.flags.writeable = False
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    start = time.time()
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()
    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):

        color = COLORS[int(classid) % len(COLORS)]
        if classid == 0:
            x,y,w,h = box
            h=y+h
            w=x+w
            # print(image.shape)
            # print(h)
            # crop = image[int(y):h,int(x):w,:]
            # print(crop.shape)
            # print(box)
            with mpPose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                #Media pose prediction ,we are 
                results = pose.process(image[int(y):h,int(x):w,:])

            #Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing 
            mpDraw.draw_landmarks(image[int(y):h,int(x):w,:], results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                ) 
        # cv2_imshow(image)

        # writing in the video file 
        # out.write(image)
    end_drawing = time.time()
    # Code to quit the video incase you are using the webcam
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000) 
    cv2.putText(image, fps_label,(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.imshow('Activity recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()