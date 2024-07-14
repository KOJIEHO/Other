import cv2
import time


# Выбираем трекер для OpenCV
tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture('video.mp4')
ok, frame = video.read()

# Выбор таргета
# bbox = cv2.selectROI(frame)

# Поиск таргета
wsize, hsize = (250, 100)
img_rgb = frame
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, (250, 100), interpolation=cv2.INTER_LINEAR)
result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
val_min, val_max, min_loc, max_loc = cv2.minMaxLoc(result)
x1 = max_loc[0]
y1 = max_loc[1]
x2 = wsize
y2 = hsize
cv2.rectangle(img_rgb, (x1, y1), (x1 + x2, y1 + y2), 255, 2)
# cv2.imwrite('res.jpg', img_rgb)
bbox = (x1, y1, x2, y2)

count = 0
time1 = time.time()
ok = tracker.init(frame, bbox)
while True:
    ok, frame = video.read()
    ok, bbox= tracker.update(frame)
    
    (x, y, w, h) = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2, 1)

    time2 = time.time()
    if (time2 - time1 >= 1):
        cv2.imwrite(f'ResKCF/res{count}.jpg', frame)
        time1 = time2
        count += 1

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == "q":
        break
cv2.destroyAllWindows()
