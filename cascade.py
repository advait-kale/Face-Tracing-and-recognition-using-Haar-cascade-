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
save_dir = r"C:\Advait\VS_Code\VS code 2.0\Face recognition\Images"
known_path = os.path.join(save_dir, "known")
runtime_path = os.path.join(save_dir, "runtime_images")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
def take_images():
    print("Taking Images")
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
            file_name = os.path.join(known_path, f"{temp_name}_{count}.jpeg")
            cv2.imwrite(file_name, face)
            count += 1

        cv2.imshow('img',img) #Displays the current frame (img) in a window titled 'img'
        k = cv2.waitKey(30) & 0xff
        if(k == 27 or count >= 100):#Checks if the pressed key is ESC (ASCII value 27).
            break

    cap.release()#Releases the webcam resource.
    cv2.destroyAllWindows()
    return True
def analyse_img():
    print("Analysing Image")
    while True:
        if os.path.exists(save_dir):
            # Get all known images
            known_images = [os.path.join(known_path, f) for f in os.listdir(known_path) if f.endswith(".jpeg")]
            # Get all runtime images
            runtime_images = [os.path.join(runtime_path, f) for f in os.listdir(runtime_path) if f.endswith(".jpeg")]
            while True:
                count = 0 #number of the saved images
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
                    file_name = os.path.join(runtime_path, f"{temp_name}_{count}.jpg")
                    count += 1
                    cv2.imwrite(file_name, face)
                    # Compare runtime images with known images
                    for r_img in runtime_images:
                        print(f"\nChecking {os.path.basename(r_img)} against known images...")
                        for k_img in known_images:
                            try:
                                result = DeepFace.verify(img1_path=r_img, img2_path=k_img, model_name="VGG-Face")
                                if result["verified"]:
                                    print(f"Match found: {os.path.basename(r_img)} matches {os.path.basename(k_img)}")
                                    break
                            except Exception as e:
                                print(f"Error comparing {r_img} and {k_img}: {e}")
                        cv2.imshow("Analyse", img)
                if cv2.waitKey(30) & 0xff == 27:  # ESC to stop
                    break

            cap.release()   
            cv2.destroyAllWindows()

def raise_error():
    raise Exception("Error")

def main():
    temp_button = input("Enter 1 to run ")
    if(temp_button == '1'):
        print("Hi Temp this will beocome a seperate winow to woth buttons")
        return_value_flag = take_images()
        if(return_value_flag):
            analyse_img()

main()
