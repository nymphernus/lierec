<body>
<h2>Описание проекта</h2>
<p>Данный проект реализует систему отслеживания зрачков, лиц и позы человека с использованием библиотек OpenCV и MediaPipe. Он позволяет обрабатывать видеофайлы, извлекая координаты ключевых точек зрачков, лиц и позы, и сохранять их в текстовые файлы.</p>

<h2>Установка</h2>
<ol>
    <li>Python 3.10.10</li>
    <li>Установка Cmake необходима для dlib</li>
        <pre><code>https://cmake.org/download/</code></pre>
    <li>Клонируйте репозиторий:
        <pre><code>git clone https://github.com/nymphernus/lierec.git</code></pre>
    </li>
    <li>Установите необходимые библиотеки:
        <pre><code>python -m pip install -r requirements.txt</code></pre>
    </li>
    <li>Замените файл-пустышку на готовую модель с Google Drive:
        <pre><code>/pupil_tracker/trained_models/shape_predictor_68_face_landmarks.dat</code></pre>
        <p>Trained model - https://drive.google.com/file/d/1jYe-izvmbfGyRE_4Ww-XbVrk9HpmQAmV/view?usp=share_link</p>
    </li>
</ol>

<h2>Использование</h2>
<ol>
    <li>Запустите скрипт:
        <pre><code>start.bat</code></pre>
    </li>
    <li>В появившемся диалоговом окне выберите видеофайл для обработки.</li>
    <li>Скрипт автоматически начнет обработку видео, создавая временные файлы для каждого типа отслеживаемых данных.</li>
    <li>После завершения обработки будут созданы файлы с координатами:
        <ul>
            <li><code>*_pupil_complete.txt</code> — координаты зрачков</li>
            <li><code>*_face_complete.txt</code> — координаты лицевых точек</li>
            <li><code>*_pose_complete.txt</code> — координаты ключевых точек позы</li>
        </ul>
    </li>
</ol>
<h2>Функциональность</h2>
<ul>
    <li><strong>Отслеживание зрачков</strong>: Записывает координаты зрачков в файл.</li>
    <li><strong>Отслеживание лиц</strong>: Записывает координаты ключевых точек лица.</li>
    <li><strong>Отслеживание позы</strong>: Записывает координаты ключевых точек позы.</li>
    <li><strong>Многопоточность</strong>: Обработка видеофайлов осуществляется с использованием многопоточности для повышения производительности.</li>
</ul>
<h2>Скриншот</h2>
<img src="https://user-images.githubusercontent.com/103174654/200115061-cb617eb5-a75a-4a38-9e60-2e318cfc9af6.jpg" alt="img_1">
<h2>Примечания</h2>
<ul>
    <li>Убедитесь, что видеофайл имеет формат, поддерживаемый OpenCV (например, .mp4).</li>
    <li>Обработка может занять некоторое время в зависимости от длины видео и количества доступных потоков.</li>
</ul>
</body>
