from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import cv2 as cv


# Создаем аргументы - используем видеокамеру и метод csrt
ap = argparse.ArgumentParser()
ap.add_argument("-v")
ap.add_argument("-t", "--tracker", type=str, default="kcf")
args = vars(ap.parse_args())

# "csrt": cv2.TrackerCSRT_create,
# "kcf": cv2.TrackerKCF_create,
# "boosting": cv2.TrackerBoosting_create,
# "mil": cv2.TrackerMIL_create,
# "tld": cv2.TrackerTLD_create,
# "medianflow": cv2.TrackerMedianFlow_create,
# "mosse": cv2.TrackerMOSSE_create

# Выбираем трекер для OpenCV
tracker = cv2.TrackerKCF_create()

# Координаты рамки объекта слежения
initBB = None

# Запуск видеокамеры
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Перебор кадров из видеопотока
while True:
    # Получаем видеопоток, выделяем первый кадр, обрезаем его для быстроты работы, запоминаем его размеры
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    # Проверка на отслеживание объекта
    if initBB is not None:
        # Обновляем информацию об объекте слежения на кадре
        (success, box) = tracker.update(frame)
        # Если удалось отследить, то обновляем координаты рамки
        if success:
            (x, y, w, h) = [int(v) for v in box]
            print('qwer', (x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Обновляем информацию о fps
        fps.update()
        fps.stop()
        # Информация на кадре
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # Отображение информации
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Показываем выходной кадр
    cv2.imshow("Frame", frame)
    # Переменная, отвечающая за кнопки
    key = cv2.waitKey(1) & 0xFF
    # w - Выбор рамки объекта слежения (SPACE - подтвердить действие)
    # q - Закрыть программу
    if key == ord("w"):
        #### Это вариант с выбором объекта с помощью мышки ####
        # initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        #### Вариант с поиском объекта на первом кадре ####
        img_rgb = frame
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = cv.imread('target.jpg', cv.IMREAD_GRAYSCALE)
        template = cv2.resize(template, (100, 100), interpolation=cv2.INTER_LINEAR)
        w, h = template.shape[::-1]

        result = cv.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
        val_min, val_max, min_loc, max_loc = cv2.minMaxLoc(result)
        t_left = max_loc
        b_right = (t_left[0] + w, t_left[1] + h)

        cv.rectangle(img_rgb, t_left, b_right, 255, 2)
        cv.imwrite('res.jpg', img_rgb)
        x, y = t_left[0], t_left[1]
        initBB = (x, y, x + w, y + h)

        # Запускаем работать трекер и счетчик fps
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif key == ord("q"):
        break
# Выключаем камеру после закрытия программы
if not args.get("video", False):
    vs.stop()
# Зарываем все окна
cv2.destroyAllWindows()
