import cv2
import mediapipe as mp
from pupil_tracker import pupilTracking as pt
from tkinter import filedialog as fd
import multiprocessing as mpr
import math
import time
import os

# создание экземпляра класса для отслеживания зрачков
spy = pt()

# создание экземпляра класса для отслеживания лиц
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
# индексы отслеживаемых точек на лице
face_lm_id = (1, 6, 23, 27, 130, 243, 253, 257, 359, 463)

# создание экземпляра класса для отслеживания поз
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# отслеживание точек на теле
def pose_rec(cropped_img, output_fileName_pose):
    """
    Записывает координаты ключевых точек позы в файл.
    cropped_img: изображение, на котором обнаруживается поза
    output_fileName_pose: имя выходного файла для записи координат
    """
    # обработка изображения для получения ключевых точек позы
    res = pose.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        with open(output_fileName_pose, 'a') as f:
            # перебор всех ключевых точек позы
            for _, lm in enumerate(res.pose_landmarks.landmark):
                # получение высоты и ширины обработанного изображения
                h, w, _ = cropped_img.shape
                # вычисление координаты точки на изображении
                cx, cy = int(lm.x * w), int(lm.y * h)
                f.write(f"{cx} {cy} ")
            f.write('\n')

# отслеживание зрачков
def pupil_rec(img, output_fileName_pupil):
    """
    Записывает координаты зрачков в файл.
    img: изображение, на котором обнаруживаются зрачки
    output_fileName_pupil: имя выходного файла для записи координат
    """
    spy.refresh(img)
    if spy.pupils_located:
        x_left, y_left = spy.pupil_left_coords()
        x_right, y_right = spy.pupil_right_coords()
    else:
        x_left, y_left = 0, 0
        x_right, y_right = 0, 0

    with open(output_fileName_pupil, 'a') as f:
        f.write(f"{x_left} {y_left} {x_right} {y_right}\n")


# отслеживание точек на лице
def face_rec(img, output_fileName_face):
    """
    Записывает координаты ключевых точек лица в файл.
    img: изображение, на котором обнаруживаются ключевые точки лица
    output_fileName_face: имя выходного файла для записи координат
    """
    # обработка изображения для получения ключевых точек лица
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # проверка наличия ключевых точек лица в обработанном изображении
    if res.multi_face_landmarks:
        # перебор всех обнаруженных ключевых точек лица
        for face_lms in res.multi_face_landmarks:
            with open(output_fileName_face, 'a') as f:
                # перебор всех идентификаторов и координат ключевых точек лица
                for id, lm in enumerate(face_lms.landmark):
                    # получение высоты и ширины обработанного изображения
                    ih, iw, _ = img.shape
                    # вычисление координаты точки на изображении
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # проверка, является ли ключевая точка лицом
                    if id in face_lm_id:
                        # запись координаты ключевой точки в файл
                        f.write(f"{x} {y} ")
                f.write('\n')

def process_video_multiprocessing(thread_list):
    # получаем количество процессов и имя файла из списка потоков
    num_processes, fileName = thread_list
    # открываем видеофайл и получаем количество кадров в нем
    cap = cv2.VideoCapture(fileName)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # вычисляем количество кадров, которые будет обрабатывать каждый процесс  
    frames_on_process = math.ceil(frame_count/mpr.cpu_count())
    # вычисляем начальный и конечный кадры для текущего процесса
    start_frame = frames_on_process*num_processes
    end_frame = frames_on_process*(num_processes+1)

    # создаем временные файлы для записи результатов
    output_fileName_pupil = fileName.replace(".mp4", f"_{num_processes:02d}_pupil_temp.txt")
    output_fileName_face = fileName.replace(".mp4", f"_{num_processes:02d}_face_temp.txt")
    output_fileName_pose = fileName.replace(".mp4", f"_{num_processes:02d}_pose_temp.txt")
    
    open(output_fileName_pupil, 'w')
    open(output_fileName_face, 'w')
    open(output_fileName_pose, 'w')
    
    count = start_frame

    try:
        while count < end_frame:
            success, img = cap.read()
            if not success:
                break
            # устанавливаем позицию чтения видеофайла на текущий кадр
            cap.set(1, start_frame)
            img = cv2.resize(img,(1920,1080))
            cropped_img = cv2.rotate(cv2.resize(img[10:350, 1285:1900],(1920,1080)), 2)

            # выполняем обработку кадра
            pupil_rec(img, output_fileName_pupil)
            face_rec(img, output_fileName_face)
            pose_rec(cropped_img, output_fileName_pose)
            
            count+=1
    except Exception as e:
        print(f"Ошибка: {e}")
        cap.release()
    finally:
        cap.release()

def files_combine(search_mask, switch_mask):
    # получаем директорию файла и исходный список файлов
    dir = os.path.abspath(fileName).replace(os.path.basename(fileName), '')
    output_list = [i for i in os.listdir(dir) if i.endswith(search_mask)]
    # сортируем список файлов
    output_list.sort()

    # формируем имя выходного файла
    output_fileName = fileName.replace(".mp4",switch_mask)
    with open(output_fileName,'w') as f:
        # читаем содержимое файла и записываем его в выходной файл
        for j in output_list:
            f.write(open(os.path.join(dir, j)).read())
            # удаляем прочитанный файл
            os.remove(os.path.join(dir, j))

def multi_process():
    # создаем пул процессов, где num_processe = кол-во потоков процессора
    p = mpr.Pool(num_processes)
    # запускаем каждый экземпляр функции на отдельном потоке
    p.map(process_video_multiprocessing, thread_list)

    # объединяем временные файлы
    files_combine('_pupil_temp.txt','_pupil_complete.txt')
    files_combine('_face_temp.txt','_face_complete.txt')
    files_combine('_pose_temp.txt','_pose_complete.txt')


if __name__ == '__main__':
    # выбор видеофайла через диалоговое окно
    fileName = fd.askopenfilename()
    # определение кол-ва доступных потоков на компьютере
    num_processes = mpr.cpu_count()

    # создание списка потоков, каждый из которых будет обрабатывать свою часть видеофайла
    thread_list = [[i, fileName] for i in range(num_processes)]

    start_time = time.time()

    # запуск обработки видеофайла в нескольких потоках
    multi_process()

    finish_time = math.ceil((time.time() - start_time))
    print(f"\n Обработка закончена\n Видеофайл - {fileName}\n Потоков - {num_processes}\n Время - {finish_time} сек.")
