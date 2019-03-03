import dlib
import numpy as np
import cv2
import time
import os
from tkinter import messagebox


save_file_path = os.path.join(os.getcwd(),"saved_face")
data_path = os.path.join(os.getcwd(),"data")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.getcwd(),"data","shape_predictor_68_face_landmarks.dat"))
cap = cv2.VideoCapture(0)
cap.set(3, 400)
cap.set(4, 300)
print(cap.get(3),cap.get(4))

while cap.isOpened():
    ret, img = cap.read()
    cv2.imshow("show emotion on your face, press s to generate sticker", img)
    user_input = cv2.waitKey(1)
    if user_input == ord('r'):
        # file_name = "face_%s.jpg" % time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        # cv2.imwrite(os.path.join(save_file_path,file_name), img)
        # print("file saved to %s..." % os.path.join(save_file_path,file_name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 1)
        print("there are %d faces" % len(faces))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(faces) > 0:
            cv2.putText(img, "Face recognized: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            for face in faces:
                shape = predictor(img, face)
                for pt in shape.parts():
                    pt_pos = (pt.x, pt.y)
                    cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
                cv2.imshow("image", img)
        else:
            messagebox.showinfo("Sad", "NO face recognized!!")
    if user_input == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()