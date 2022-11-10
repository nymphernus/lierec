import cv2
import mediapipe as mp
from tkinter import filedialog as fd
from pupil_tracker import pupilTracking as pt
import time

fileName = fd.askopenfilename()
cap = cv2.VideoCapture(fileName)
output_fileName_pose = fileName.replace(".mp4","_pose.txt")
output_fileName_pupil = fileName.replace(".mp4","_pupil.txt")
output_fileName_face = fileName.replace(".mp4","_face.txt")

start_time = time.time()
spy = pt()
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def poseRec(cropped_img):
    res = pose.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    with open(output_fileName_pose, 'a') as f:
        print('', file=f)
    if res.pose_landmarks:
        mpDraw.draw_landmarks(cropped_img, res.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(res.pose_landmarks.landmark):
            h, w, c = cropped_img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            with open(output_fileName_pose, 'a') as f:
                print(''+f'{cx},'+ f'{cy} ', file=f, end='')
            cv2.circle(cropped_img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
    return cropped_img

def pupilRec(img):
    spy.refresh(img)
    if spy.pupils_located:
        x_left, y_left = spy.pupil_left_coords()
        x_right, y_right = spy.pupil_right_coords()
    else:
        x_left, y_left = 0, 0
        x_right, y_right = 0, 0

    with open(output_fileName_pupil, 'a') as f:
        print('', file=f)
        print( f'{x_left},{y_left} '+ f'{x_right},{y_right}', file=f, end='')
    return spy.annotated_frame()

def faceRec(img):
    res = faceMesh.process(img)
    with open(output_fileName_face, 'a') as f:
        print('', file=f)
    if res.multi_face_landmarks:
        for faceLms in res.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
        for id, lm in enumerate(faceLms.landmark):
            ih, iw, ic = img.shape
            x, y = int(lm.x*iw), int(lm.y*ih)
            with open(output_fileName_face, 'a') as f:
                print(''+f'{x},'+ f'{y} ', file=f, end='')
    return img

with open(output_fileName_pose, 'w') as f:
        pass
with open(output_fileName_pupil, 'w') as f:
        pass
with open(output_fileName_face, 'w') as f:
        pass

tactCounter = 0
while True:
    success, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img,(1920,1080))
    img_face = img
    cropped_img = cv2.rotate(cv2.resize(img[10:350, 1285:1900],(1920,1080)), 2)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pupilRec(img)
    faceRec(img_face)
    poseRec(cropped_img)
    # cv2.namedWindow("pupil", cv2.WINDOW_NORMAL)
    # cv2.imshow("pupil", cv2.resize(pupilRec(img), (1280, 720)))
    # cv2.imshow("face", cv2.resize(faceRec(img_face), (640, 360)))
    # cv2.imshow("pose", cv2.resize(poseRec(cropped_img), (360, 640)))
    cv2.imshow("counter",tactCounter)
    print(tactCounter," - %s seconds" % (time.time() - start_time))
    tactCounter+=1
    if (cv2.waitKey(1) == 27) or (cv2.waitKey(25) == ord('q')):
        break
print("%s seconds" % (time.time() - start_time))
cap.release()
cv2.destroyAllWindows()
