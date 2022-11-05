import cv2
import mediapipe as mp
from tkinter import filedialog as fd
from pupil_tracker import pupilTracking as pt


spy = pt()
fileName = fd.askopenfilename()
cap = cv2.VideoCapture(fileName)
output_fileName = fileName.replace("mp4","txt")


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

with open(output_fileName, 'w') as f:
        pass
count = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_img = img[130:320, 1350:1750]
    
    res = pose.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    with open(output_fileName, 'a') as f:
        print('', file=f)
        print( f'{count} ', file=f, end='')
    if res.pose_landmarks:
        mpDraw.draw_landmarks(cropped_img, res.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(res.pose_landmarks.landmark):
            h, w, c = cropped_img.shape
            
            cx, cy = int(lm.x * w), int(lm.y * h)
            with open(output_fileName, 'a') as f:
                print('('+f'{cx},'+ f'{cy});', file=f, end='')
            cv2.circle(cropped_img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    
    spy.refresh(img)
    img = spy.annotated_frame()

    count+=1
    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    cv2.imshow("crop", cv2.resize(cv2.rotate(cropped_img, 2), (360, 640)))
    cv2.imshow("face", cv2.resize(img, (1280, 720)))
    if cv2.waitKey(1) == 27:
        break
