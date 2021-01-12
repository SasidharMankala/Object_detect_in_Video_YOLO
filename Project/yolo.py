import numpy as np
import cv2

img = cv2.imread('sasi.jpg')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cropimg = roi_color
cv2.imshow('img',roi_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
