import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from pathlib import Path
import re

class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Визуализация данных отслеживания")
        self.root.geometry("1200x800")
        
        # Переменные
        self.data_files = {}
        self.current_file = tk.StringVar()
        self.plot_type = tk.StringVar(value="line")
        self.selected_landmarks = []
        
        # Создание интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        # Верхняя панель управления
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Выбор файла
        ttk.Button(control_frame, text="Загрузить файл", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Файл:").pack(side=tk.LEFT, padx=(20, 5))
        file_combo = ttk.Combobox(control_frame, textvariable=self.current_file, 
                                 state="readonly", width=30)
        file_combo.pack(side=tk.LEFT, padx=5)
        file_combo.bind('<<ComboboxSelected>>', self.on_file_selected)
        
        # Тип графика
        ttk.Label(control_frame, text="Тип:").pack(side=tk.LEFT, padx=(20, 5))
        plot_combo = ttk.Combobox(control_frame, textvariable=self.plot_type,
                                 values=["line", "scatter", "bar"], state="readonly", width=10)
        plot_combo.pack(side=tk.LEFT, padx=5)
        plot_combo.bind('<<ComboboxSelected>>', self.update_plot)
        
        # Кнопка обновления
        ttk.Button(control_frame, text="Обновить график", command=self.update_plot).pack(side=tk.LEFT, padx=20)
        
        # Основная область
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Левая панель (элементы управления)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Выбор точек для отображения
        ttk.Label(left_frame, text="Точки для отображения:").pack(anchor=tk.W, pady=(0, 5))
        
        self.points_frame = ttk.Frame(left_frame)
        self.points_frame.pack(fill=tk.Y, expand=True)
        
        # Холст для чекбоксов с прокруткой
        canvas_frame = ttk.Frame(self.points_frame)
        canvas_frame.pack(fill=tk.Y, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, width=200)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Правая панель (график)
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas_plot = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas_plot.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Статусная строка
        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_file(self):
        """Загрузка файла с данными"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                file_name = Path(file_path).stem
                data = self.parse_data_file(file_path)
                
                if data is not None:
                    self.data_files[file_name] = {
                        'path': file_path,
                        'data': data,
                        'type': self.detect_data_type(file_name)
                    }
                    
                    # Обновление списка файлов
                    file_names = list(self.data_files.keys())
                    self.current_file.set(file_name)
                    
                    combo = self.root.nametowidget('.!frame.!frame.!combobox2')
                    combo['values'] = file_names
                    
                    # Создание чекбоксов для точек
                    self.create_point_checkboxes(data)
                    
                    # Отображение графика
                    self.update_plot()
                    
                    self.status_var.set(f"Загружен файл: {file_name}")
                else:
                    self.status_var.set("Ошибка: Неверный формат файла")
                    
            except Exception as e:
                self.status_var.set(f"Ошибка загрузки файла: {str(e)}")
    
    def parse_data_file(self, file_path):
        """Парсинг файла с данными"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
                
            # Определение типа данных по первой строке
            first_line = lines[0].strip()
            values = first_line.split()
            
            # Преобразование в числа
            numeric_data = []
            for line in lines:
                if line.strip():
                    try:
                        row = [float(x) if x != '0' else 0 for x in line.strip().split()]
                        numeric_data.append(row)
                    except ValueError:
                        continue
            
            if not numeric_data:
                return None
                
            return np.array(numeric_data)
            
        except Exception as e:
            print(f"Ошибка парсинга файла: {e}")
            return None
    
    def detect_data_type(self, file_name):
        """Определение типа данных по имени файла"""
        if 'pupil' in file_name.lower():
            return 'pupil'
        elif 'face' in file_name.lower():
            return 'face'
        elif 'pose' in file_name.lower():
            return 'pose'
        else:
            return 'unknown'
    
    def create_point_checkboxes(self, data):
        """Создание чекбоксов для выбора точек"""
        # Очистка предыдущих чекбоксов
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        if data.size == 0:
            return
            
        # Определение количества точек
        num_values = len(data[0]) if len(data) > 0 else 0
        
        if self.detect_data_type(self.current_file.get()) == 'pupil':
            # Для зрачков - 4 точки (x_left, y_left, x_right, y_right)
            point_names = ['Левый зрачок X', 'Левый зрачок Y', 'Правый зрачок X', 'Правый зрачок Y']
        elif self.detect_data_type(self.current_file.get()) == 'face':
            # Для лица - точки по индексам
            point_names = [f'Точка {i//2}' for i in range(0, min(num_values, 20), 2)]
        elif self.detect_data_type(self.current_file.get()) == 'pose':
            # Для позы - точки тела
            point_names = [f'Точка тела {i//2}' for i in range(0, min(num_values, 66), 2)]
        else:
            point_names = [f'Значение {i}' for i in range(num_values)]
        
        # Создание чекбоксов
        self.point_vars = []
        for i, name in enumerate(point_names):
            if i * 2 < num_values:  # Проверка, что индекс существует
                var = tk.BooleanVar(value=True)
                self.point_vars.append((var, i * 2))  # Сохраняем переменную и индекс
                
                cb = ttk.Checkbutton(self.scrollable_frame, text=name, variable=var)
                cb.pack(anchor=tk.W, pady=2)
        
        # Кнопка "Выбрать все"
        select_all_var = tk.BooleanVar(value=True)
        
        def toggle_all():
            state = select_all_var.get()
            for var, _ in self.point_vars:
                var.set(state)
            self.update_plot()
        
        select_all_cb = ttk.Checkbutton(
            self.scrollable_frame, 
            text="Выбрать все", 
            variable=select_all_var,
            command=toggle_all
        )
        select_all_cb.pack(anchor=tk.W, pady=(10, 2))
    
    def get_selected_points(self):
        """Получение выбранных точек"""
        selected = []
        if hasattr(self, 'point_vars'):
            for var, index in self.point_vars:
                if var.get():
                    selected.append(index)
        return selected
    
    def on_file_selected(self, event=None):
        """Обработчик выбора файла"""
        file_name = self.current_file.get()
        if file_name in self.data_files:
            data = self.data_files[file_name]['data']
            self.create_point_checkboxes(data)
            self.update_plot()
    
    def update_plot(self, event=None):
        """Обновление графика"""
        try:
            file_name = self.current_file.get()
            if not file_name or file_name not in self.data_files:
                return
                
            data = self.data_files[file_name]['data']
            if data.size == 0:
                return
                
            # Очистка графика
            self.ax.clear()
            
            # Получение выбранных точек
            selected_points = self.get_selected_points()
            if not selected_points:
                selected_points = list(range(0, min(len(data[0]), 20), 2))  # По умолчанию первые 10 точек
            
            # Тип графика
            plot_type = self.plot_type.get()
            
            # Создание графика
            x_axis = range(len(data))  # Номера кадров
            
            for point_idx in selected_points:
                if point_idx < len(data[0]):
                    y_data = data[:, point_idx]
                    
                    # Получение имени точки
                    if self.detect_data_type(file_name) == 'pupil':
                        names = ['Левый X', 'Левый Y', 'Правый X', 'Правый Y']
                        name = names[point_idx] if point_idx < len(names) else f'Точка {point_idx}'
                    elif self.detect_data_type(file_name) == 'face':
                        name = f'Лицо {point_idx//2}'
                    elif self.detect_data_type(file_name) == 'pose':
                        name = f'Поза {point_idx//2}'
                    else:
                        name = f'Значение {point_idx}'
                    
                    # Построение графика
                    if plot_type == "line":
                        self.ax.plot(x_axis, y_data, label=name, linewidth=1)
                    elif plot_type == "scatter":
                        self.ax.scatter(x_axis, y_data, label=name, s=1)
                    elif plot_type == "bar":
                        # Для bar графика показываем только первые 50 точек
                        show_points = min(50, len(x_axis))
                        self.ax.bar(x_axis[:show_points], y_data[:show_points], label=name, alpha=0.7)
            
            # Настройка графика
            self.ax.set_xlabel('Номер кадра')
            self.ax.set_ylabel('Значение')
            self.ax.set_title(f'Данные: {file_name}')
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax.grid(True, alpha=0.3)
            
            # Обновление холста
            self.canvas_plot.draw()
            
            self.status_var.set(f"Отображено {len(selected_points)} точек из файла {file_name}")
            
        except Exception as e:
            self.status_var.set(f"Ошибка построения графика: {str(e)}")
    
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

def main():
    root = tk.Tk()
    app = DataVisualizerApp(root)
    app.run()

if __name__ == "__main__":
    main()