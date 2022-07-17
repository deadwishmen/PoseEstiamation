from cProfile import label
from tkinter import Frame
from unittest import result
import cv2
import mediapipe as mp
import pandas as pd

# khoi tạo thư viện mediapipe

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "BODYSWING"
no_of_frames = 600

# đọc ảnh từ web cam
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(result.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm
def draw_landmark_on_image(mpDraw, results, img):
    # vẽ các đường nối
    mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # vẽ các điểm nối
    for id, lm in enumerate(result.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0,0,255), cv2.FILLED)
    return img
# cap = cv2.VideoCapture("108.mp4")
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if ret:
        # nhận diện
        frameRGB =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frameRGB) # Convert the BGR image to RGB before processing
        if result.pose_landmarks:
            # ghi nhận thông số khung xương
            lm = make_landmark_timestep(result)
            # vẽ khung lên ảnh
            frame = draw_landmark_on_image(mpDraw, result, frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()