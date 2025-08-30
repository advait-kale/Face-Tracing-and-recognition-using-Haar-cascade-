import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(r"c:\Advait\VS_Code\VS code 2.0\Face recognition\haarcascade_frontalface_default.xml")

#variable declaration
global count #couning the images 
global name
global temp_name

def input_name():
    name = input("Enter name ")
    return name

temp_name = input_name()

cap = cv2.VideoCapture(0)
save_dir = r"C:\Advait\VS_Code\VS code 2.0\Face recognition\Images\known"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
def take_images():
    count = 0
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            #int arguments are the img then starting point of the rectange then the ending point of the rectange
            #then the color of the rectange then the width of the rectangle

            #extract face data
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100,100))
            file_name = os.path.join(save_dir, f"{temp_name}_{count}.jpg")
            cv2.imwrite(file_name, face)
            count += 1

        cv2.imshow('img',img) #Displays the current frame (img) in a window titled 'img'
        k = cv2.waitKey(30) & 0xff
        if(k == 27 or count >= 100):#Checks if the pressed key is ESC (ASCII value 27).
            break

    cap.release()#Releases the webcam resource.
    cv2.destroyAllWindows()
def analyse_img():
    print("Analysing Image")
    for i in range(100):
        file_path = os.path.join(save_dir,f"{input_name()}_{i}.jpeg")
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            result = DeepFace.analyze(actions="Gender")
        else:
            raise_error()
    print(result)

def raise_error():
    raise Exception("Error")

def main():
    temp_button = input("Enter 1 to run ")
    if(temp_button == '1'):
        print("Hi Temp this will beocome a seperate winow to woth buttons")
        input_name()
        take_images()

main()
