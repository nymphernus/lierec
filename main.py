import cv2
import mediapipe as mp
from pupil_tracker import pupilTracking as pt
from tkinter import filedialog as fd
import multiprocessing as mpr
import math
import time
import os

spy = pt()
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def poseRec(cropped_img, output_fileName_pose):
    res = pose.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        for id, lm in enumerate(res.pose_landmarks.landmark):
            h, w, c = cropped_img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            with open(output_fileName_pose, 'a') as f:
                print(''+f'{cx} '+ f'{cy} ', file=f, end='')
    with open(output_fileName_pose, 'a') as f:
            print('', file=f)
def pupilRec(img, output_fileName_pupil):
    spy.refresh(img)
    if spy.pupils_located:
        x_left, y_left = spy.pupil_left_coords()
        x_right, y_right = spy.pupil_right_coords()
    else:
        x_left, y_left = 0, 0
        x_right, y_right = 0, 0

    with open(output_fileName_pupil, 'a') as f:
        print( f'{x_left} {y_left} '+ f'{x_right} {y_right}', file=f)

def faceRec(img, output_fileName_face):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = faceMesh.process(imgRGB)
    if res.multi_face_landmarks:
        for faceLms in res.multi_face_landmarks:
            pass
        for id, lm in enumerate(faceLms.landmark):
            ih, iw, ic = img.shape
            x, y = int(lm.x*iw), int(lm.y*ih)
            with open(output_fileName_face, 'a') as f:
                print(''+f'{x} '+ f'{y} ', file=f, end='')
    with open(output_fileName_face, 'a') as f:
        print('', file=f)

def process_video_multiprocessing(multiprocfile):
    num_processes, fileName = multiprocfile
    cap = cv2.VideoCapture(fileName)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_on_process = math.ceil(frame_count/mpr.cpu_count())
    start_frame = frames_on_process*num_processes
    end_frame = frames_on_process*(num_processes+1)
    
    if num_processes > 9:
        output_fileName_pupil = fileName.replace(".mp4","_{}_pupil_temp.txt".format(num_processes))
        output_fileName_face = fileName.replace(".mp4","_{}_face_temp.txt".format(num_processes))
        output_fileName_pose = fileName.replace(".mp4","_{}_pose_temp.txt".format(num_processes))
    else:
        output_fileName_pupil = fileName.replace(".mp4","_0{}_pupil_temp.txt".format(num_processes))
        output_fileName_face = fileName.replace(".mp4","_0{}_face_temp.txt".format(num_processes))
        output_fileName_pose = fileName.replace(".mp4","_0{}_pose_temp.txt".format(num_processes))
    
    with open(output_fileName_pupil, 'w') as f:
        pass
    with open(output_fileName_face, 'w') as f:
        pass
    with open(output_fileName_pose, 'w') as f:
        pass
    
    count = start_frame
    
    try:
        while count < end_frame:
            success, img = cap.read()
            if img is None:
                break
            cap.set(1, start_frame)
            img = cv2.resize(img,(1920,1080))
            cropped_img = cv2.rotate(cv2.resize(img[10:350, 1285:1900],(1920,1080)), 2)

            pupilRec(img, output_fileName_pupil)
            faceRec(img, output_fileName_face)
            poseRec(cropped_img, output_fileName_pose)
            
            count+=1
    except:
        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()
def files_combine(search_mask, switch_mask):
    output_list = [i for i in os.listdir() if i.endswith(search_mask)]
    output_list.sort()
    output_fileName = fileName.replace(".mp4",switch_mask)
    with open(output_fileName,'w') as f:
        for j in output_list:
            s = open(j).read()
            f.write(s)
            os.remove(j)
def multi_process():
    p = mpr.Pool(num_processes)
    p.map(process_video_multiprocessing, multiprocfile)
    files_combine('_pupil_temp.txt','_pupil_complete.txt')
    files_combine('_face_temp.txt','_face_complete.txt')
    files_combine('_pose_temp.txt','_pose_complete.txt')


if __name__ == '__main__':
    fileName = fd.askopenfilename()
    num_processes = mpr.cpu_count()
    multiprocfile = [[0] * 2 for i in range(num_processes)]
    for nop in range(num_processes):
        multiprocfile[nop][0] = nop
        multiprocfile[nop][1] = fileName
    start_time = time.time()
    multi_process()
    print("%s seconds" % math.ceil((time.time() - start_time)))
