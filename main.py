import cv2
from cv2 import rectangle
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog as fd
from pupil_tracker import pupilTracking as pt


spy = pt()
fileName = fd.askopenfilename()
cap = cv2.VideoCapture(fileName)
output_fileName = fileName.replace("mp4","txt")

# cap = cv2.VideoCapture(0)
# output_fileName = "vcam.txt"

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
 
pTime = 0
cTime = 0

def handsRec(img):
    res = hands.process(img)
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
def faceRec(img):
    res = faceMesh.process(img)
    if res.multi_face_landmarks:
        for faceLms in res.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
        for id, lm in enumerate(faceLms.landmark):
            # print(lm)
            ih, iw, ic = img.shape
            x, y = int(lm.x*iw), int(lm.y*ih)
            # print(id, x, y)

def poseRec(imgRGB, img):
    res = pose.process(imgRGB)
    with open(output_fileName, 'w') as f:
        print('', file=f)
        print( f'{count} ', file=f, end='')
    if res.pose_landmarks:
        mpDraw.draw_landmarks(img, res.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(res.pose_landmarks.landmark):
            h, w, c = img.shape
            
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            with open(output_fileName, 'w') as f:
                print('('+f'{cx},'+ f'{cy});', file=f, end='')
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

def spyRec(img):
    spy.refresh(img)
    img = spy.annotated_frame()
    return img

def secDots():
    cv2.circle(img, (1282, 357), 5, (0, 0, 100), cv2.FILLED)
    cv2.circle(img, (1920, 5), 5, (0, 0, 100), cv2.FILLED)
    cv2.circle(img, (1282, 5), 5, (0, 0, 100), cv2.FILLED)
    cv2.circle(img, (1920, 357), 5, (0, 0, 100), cv2.FILLED)
    cv2.rectangle(img, (1282, 357),(1920, 5), (255,0,0), 2)
    area = (1282, 5, 1920, 357)

count = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_img = img[130:320, 1350:1750]
    cropped_imgRGB = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    # handsRec(imgRGB)
    # faceRec(imgRGB)
    # poseRec(imgRGB, img)
    poseRec(cropped_imgRGB, cropped_img)
    # spyRec(img)
    
    img = spyRec(img)
    text = ""

    if spy.is_blinking():
        text = "Blinking"
    elif spy.is_right():
        text = "Looking right"
    elif spy.is_left():
        text = "Looking left"
    elif spy.is_center():
        text = "Looking center"
    
    cv2.putText(img, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = spy.pupil_left_coords()
    right_pupil = spy.pupil_right_coords()
    cv2.putText(img, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(img, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # secDots()
    
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
 

    count+=1
    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    cv2.imshow("crop", cv2.resize(cv2.rotate(cropped_img, 2), (360, 640)))
    cv2.imshow("face", cv2.resize(img, (1280, 720)))
    if cv2.waitKey(1) == 27:
        break
