import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(r"c:\Advait\VS_Code\VS code 2.0\Face recognition\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        #int arguments are the img then starting point of the rectange then the ending point of the rectange
        #then the color of the rectange then the width of the rectangle
    cv2.imshow('img',img)#Displays the current frame (img) in a window titled 'img'
    k = cv2.waitKey(30) & 0xff
    if(k == 27):#Checks if the pressed key is ESC (ASCII value 27).
        break
cap.release()#Releases the webcam resource.
cv2.destroyAllWindows()


