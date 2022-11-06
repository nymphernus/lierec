import cv2
import mediapipe as mp
from tkinter import filedialog as fd
from pupil_tracker import pupilTracking as pt


fileName = fd.askopenfilename()
cap = cv2.VideoCapture(fileName)
output_fileName = fileName.replace(".mp4","_pose.txt")
output_fileName_2 = fileName.replace(".mp4","_gaze.txt")

spy = pt()
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def poseRec(cropped_img):
    res = pose.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    with open(output_fileName, 'a') as f:
        print('', file=f)
        print( f'{tactCounter} ', file=f, end='')
    if res.pose_landmarks:
        mpDraw.draw_landmarks(cropped_img, res.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(res.pose_landmarks.landmark):
            h, w, c = cropped_img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            with open(output_fileName, 'a') as f:
                print('('+f'{cx},'+ f'{cy});', file=f, end='')
            cv2.circle(cropped_img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
    return cropped_img

def pupilRec(img):
    spy.refresh(img)
    left_pupil = spy.pupil_left_coords()
    right_pupil = spy.pupil_right_coords()

    with open(output_fileName_2, 'a') as f:
        print('', file=f)
        print( f'{tactCounter} ', file=f, end='')
        print( f'{left_pupil};'+ f'{right_pupil};', file=f, end='')
    return spy.annotated_frame()

with open(output_fileName, 'w') as f:
        pass
with open(output_fileName_2, 'w') as f:
        pass

tactCounter = 0
while True: 
    success, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img,(1920,1080))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    cv2.imshow("face", cv2.resize(pupilRec(img), (1280, 720)))
    cv2.imshow("crop", cv2.resize(cv2.rotate(poseRec(img[130:320, 1350:1750]), 2), (360, 640)))
    tactCounter+=1
    if (cv2.waitKey(1) == 27) or (cv2.waitKey(25) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
