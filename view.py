import cv2
import mediapipe as mp
from pupil_tracker import pupilTracking as pt
import numpy as np
from tkinter import filedialog as fd

class RealTimeVideoProcessor:
    
    def __init__(self):
        self.pupil_tracker = pt()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.pose_detector = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, enable_segmentation=False)
        
        self.face_landmark_ids = [1, 6, 23, 27, 130, 243, 253, 257, 359, 463]
        
        self.PUPIL_COLOR = (0, 255, 0)
        self.FACE_POINTS_COLOR = (255, 0, 0)
        self.POSE_COLOR = (0, 0, 255)

    def draw_pupils(self, img):
        try:
            self.pupil_tracker.refresh(img)
            
            if self.pupil_tracker.pupils_located:
                x_left, y_left = self.pupil_tracker.pupil_left_coords()
                x_right, y_right = self.pupil_tracker.pupil_right_coords()
                
                if x_left and y_left:
                    cv2.circle(img, (int(x_left), int(y_left)), 5, self.PUPIL_COLOR, -1)
                if x_right and y_right:
                    cv2.circle(img, (int(x_right), int(y_right)), 5, self.PUPIL_COLOR, -1)
                    
        except Exception as e:
            pass
        
        return img

    def draw_face_landmarks(self, img):
        try:
            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                h, w = img.shape[:2]
                
                for face_landmarks in results.multi_face_landmarks:
                    for idx in self.face_landmark_ids:
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(img, (x, y), 4, self.FACE_POINTS_COLOR, -1)
                            
        except Exception as e:
            pass
        
        return img

    def process_and_get_pose(self, img):
        try:
            h, w = img.shape[:2]
            if w > 1285 and h > 350:
                pose_region = img[10:350, 1285:1900].copy()
                
                results = self.pose_detector.process(cv2.cvtColor(pose_region, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=pose_region,
                        landmark_list=results.pose_landmarks,
                        connections=mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=self.POSE_COLOR, thickness=2, circle_radius=2)
                    )
                
                rotated_pose = cv2.rotate(pose_region, 2)
                return rotated_pose
            else:
                empty_img = np.zeros((340, 615, 3), dtype=np.uint8)
                return cv2.rotate(empty_img, 2)
                            
        except Exception as e:
            empty_img = np.zeros((340, 615, 3), dtype=np.uint8)
            return cv2.rotate(empty_img, 2)

    def process_frame(self, frame):
        try:
            annotated_frame = frame.copy()
            
            annotated_frame = self.draw_pupils(annotated_frame)
            annotated_frame = self.draw_face_landmarks(annotated_frame)
            
            pose_skeleton = self.process_and_get_pose(frame.copy())
            
            return annotated_frame, pose_skeleton
            
        except Exception as e:
            return frame, np.zeros((615, 340, 3), dtype=np.uint8)

def main():
    choice = input("Введите 1 для выбора видеофайла: ").strip()
    
    use_file = (choice == "1")
    video_source = 0
    video_path = None
    
    if use_file:
        try:
            video_path = fd.askopenfilename(filetypes=[("Video files", "*.mp4")])
            if not video_path:
                print("Файл не выбран. Используется камера.")
                use_file = False
            else:
                video_source = video_path
        except:
            print("Ошибка при выборе файла. Используется камера.")
            use_file = False
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеопоток")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    processor = RealTimeVideoProcessor()
    
    print("Начало обработки видео в реальном времени...")
    print("Нажмите 'q' для выхода")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Конец видео или ошибка захвата кадра")
                break
            
            processed_frame, pose_skeleton = processor.process_frame(frame)
            
            display_frame = cv2.resize(processed_frame, (960, 540))
            cv2.imshow('Main Cam', display_frame)
            
            if use_file:
                skeleton_display = cv2.resize(pose_skeleton, (200, 400))
                cv2.imshow('Cam[v2]', skeleton_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем")
    except Exception as e:
        print(f"Ошибка во время обработки: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")

if __name__ == '__main__':
    main()