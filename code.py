import numpy as np
import cv2 as cv

def pre_processing(frame) :
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)

    return blur

def detect_circles(contours) :
    filtered_contours = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 700:
            continue

        (x, y), radius = cv.minEnclosingCircle(cnt)

        circle_area = 3.1416 * (radius ** 2)
        circularity = area / circle_area

        if 0.5 < circularity <= 1.2:
            filtered_contours.append(cnt)

    return filtered_contours

def slice_object(frame, contours) :
    objects = []
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        objects.append(frame[y:y+h, x:x+w])

    return objects

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:

    ret, frame = cap.read()
 
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img_preprocessing = pre_processing(frame)
    ret,thresh = cv.threshold(img_preprocessing,120,255,cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_img = frame.copy()
    filtered_contours = detect_circles(contours)
    slice_object(frame, filtered_contours)
    cv.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)

    cv.imshow('frame', contour_img)
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()