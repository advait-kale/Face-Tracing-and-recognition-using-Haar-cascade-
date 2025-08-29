import cv2
import numpy as np
import os
face_cascade = cv2.CascadeClassifier(r"c:\Advait\VS_Code\VS code 2.0\Face recognition\haarcascade_frontalface_default.xml")

#variable declaration
global count #couning the images 
global name
cap = cv2.VideoCapture(0)
def take_images():
    while True:
        count = 0
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            #int arguments are the img then starting point of the rectange then the ending point of the rectange
            #then the color of the rectange then the width of the rectangle

            #extract face data
            face = gray[y:y+h, x:x+h]
            face = cv2.resize(face, (100,100))
            file_name = os.path.join("C:\Advait\VS_Code\VS code 2.0\Face recognition\Images\known", f"{name}_{count}.jpg")
            cv2.imwrite(file_name, face)
            count += 1

        cv2.imshow('img',img)#Displays the current frame (img) in a window titled 'img'
        k = cv2.waitKey(30) #& 0xff
        if(k == 27):#Checks if the pressed key is ESC (ASCII value 27).
            break
        elif(count >= 100):
            break
    cap.release()#Releases the webcam resource.
    cv2.destroyAllWindows()

def input_name():
    name = input("Enter name ")


def main():
    temp_button = input("Enter 1 to run ")
    if(temp_button == 1):
        input_name()
        take_images()

main()