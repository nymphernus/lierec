import cv2
import mediapipe as mp
from pupil_tracker import pupilTracking
from tkinter import filedialog as fd
import multiprocessing as mpr
import math
import time
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    # Класс для обработки видео с отслеживанием зрачков, лица и позы
    
    def __init__(self):
        # Инициализация всех детекторов
        self.pupil_tracker = pupilTracking()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2)
        self.pose_detector = mp.solutions.pose.Pose()
        
        self.face_landmark_ids = {1, 6, 23, 27, 130, 243, 253, 257, 359, 463}
        
        self.FRAME_WIDTH = 1920
        self.FRAME_HEIGHT = 1080
        self.CROP_REGION = (1285, 1900, 10, 350)  # x1, x2, y1, y2

    def process_pose(self, cropped_img: cv2.Mat, output_path: Path) -> None:
        # Обрабатывает позу и записывает координаты в файл
        try:
            results = self.pose_detector.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                with open(output_path, 'a') as f:
                    coordinates = []
                    for landmark in results.pose_landmarks.landmark:
                        h, w = cropped_img.shape[:2]
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        coordinates.extend([str(cx), str(cy)])
                    f.write(' '.join(coordinates) + '\n')
        except Exception as e:
            logger.error(f"Ошибка обработки позы: {e}")

    def process_pupils(self, img: cv2.Mat, output_path: Path) -> None:
        # Обрабатывает зрачки и записывает координаты в файл
        try:
            self.pupil_tracker.refresh(img)
            
            if self.pupil_tracker.pupils_located:
                x_left, y_left = self.pupil_tracker.pupil_left_coords()
                x_right, y_right = self.pupil_tracker.pupil_right_coords()
            else:
                x_left = y_left = x_right = y_right = 0

            with open(output_path, 'a') as f:
                f.write(f"{x_left} {y_left} {x_right} {y_right}\n")
        except Exception as e:
            logger.error(f"Ошибка обработки зрачков: {e}")

    def process_face_landmarks(self, img: cv2.Mat, output_path: Path) -> None:
        # Обрабатывает лицевые точки и записывает координаты в файл
        try:
            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                with open(output_path, 'a') as f:
                    for face_landmarks in results.multi_face_landmarks:
                        coordinates = []
                        h, w = img.shape[:2]
                        
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            if idx in self.face_landmark_ids:
                                x, y = int(landmark.x * w), int(landmark.y * h)
                                coordinates.extend([str(x), str(y)])
                        
                        f.write(' '.join(coordinates) + '\n')
        except Exception as e:
            logger.error(f"Ошибка обработки лицевых точек: {e}")

    def crop_image_region(self, img: cv2.Mat) -> cv2.Mat:
        # Обрезает изображение по заданной области
        x1, x2, y1, y2 = self.CROP_REGION
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        return cv2.rotate(resized, 2)

def process_video_segment(args: Tuple[int, str, int, int]) -> None:
    # Обрабатывает сегмент видео в отдельном процессе
    process_id, video_path, start_frame, end_frame = args
    processor = VideoProcessor()
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        video_path_obj = Path(video_path)
        base_name = video_path_obj.stem
        output_dir = video_path_obj.parent
        
        pupil_file = output_dir / f"{base_name}_{process_id:02d}_pupil_temp.txt"
        face_file = output_dir / f"{base_name}_{process_id:02d}_face_temp.txt"
        pose_file = output_dir / f"{base_name}_{process_id:02d}_pose_temp.txt"
        
        for file_path in [pupil_file, face_file, pose_file]:
            open(file_path, 'w').close()
        
        frame_count = start_frame
        while frame_count < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, img = cap.read()
            
            if not success:
                break
                
            img = cv2.resize(img, (processor.FRAME_WIDTH, processor.FRAME_HEIGHT))
            cropped_img = processor.crop_image_region(img)
            
            processor.process_pupils(img, pupil_file)
            processor.process_face_landmarks(img, face_file)
            processor.process_pose(cropped_img, pose_file)
            
            frame_count += 1
            
    except Exception as e:
        logger.error(f"Ошибка в процессе {process_id}: {e}")
    finally:
        if 'cap' in locals():
            cap.release()

def combine_temp_files(video_path: str, file_type: str) -> None:
    # Объединяет временные файлы в один
    try:
        video_path_obj = Path(video_path)
        base_name = video_path_obj.stem
        output_dir = video_path_obj.parent
        
        temp_files = list(output_dir.glob(f"{base_name}_*_{file_type}_temp.txt"))
        temp_files.sort()
        
        final_file = output_dir / f"{base_name}_{file_type}_complete.txt"
        
        with open(final_file, 'w') as output_file:
            for temp_file in temp_files:
                with open(temp_file, 'r') as f:
                    output_file.write(f.read())
                temp_file.unlink()
                
        logger.info(f"Файл {file_type} успешно объединен: {final_file}")
        
    except Exception as e:
        logger.error(f"Ошибка объединения файлов {file_type}: {e}")

def get_video_frame_count(video_path: str) -> int:
    # Получает количество кадров в видео
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    video_path = fd.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if not video_path:
        logger.info("Видео не выбрано")
        return
    
    num_processes = mpr.cpu_count()
    logger.info(f"Используется {num_processes} процессов")
    
    frame_count = get_video_frame_count(video_path)
    frames_per_process = math.ceil(frame_count / num_processes)
    
    process_args = []
    for i in range(num_processes):
        start_frame = frames_per_process * i
        end_frame = min(frames_per_process * (i + 1), frame_count)
        process_args.append((i, video_path, start_frame, end_frame))
    
    start_time = time.time()
    
    try:
        with mpr.Pool(processes=num_processes) as pool:
            pool.map(process_video_segment, process_args)
        
        for file_type in ['pupil', 'face', 'pose']:
            combine_temp_files(video_path, file_type)
            
        processing_time = time.time() - start_time
        logger.info(f"Обработка завершена успешно!")
        logger.info(f"Видео: {video_path}")
        logger.info(f"Процессов: {num_processes}")
        logger.info(f"Время: {math.ceil(processing_time)} сек.")
        
    except Exception as e:
        logger.error(f"Ошибка во время обработки: {e}")

if __name__ == '__main__':
    main()