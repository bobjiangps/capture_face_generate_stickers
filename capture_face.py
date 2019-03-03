import dlib
import numpy as np
import cv2
import time
import os
from tkinter import messagebox
from PIL import Image


save_file_path = os.path.join(os.getcwd(), "saved_face")
data_path = os.path.join(os.getcwd(), "data")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.getcwd(), "data", "shape_predictor_68_face_landmarks.dat"))
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
print(cap.get(3),cap.get(4))

while cap.isOpened():
    ret, img = cap.read()
    cv2.imshow("show emotion on your face, press s to generate sticker", img)
    user_input = cv2.waitKey(1)
    if user_input == ord('r'):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 1)
        print("there are %d faces" % len(faces))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(faces) > 0:
            try:
                height_max = 0
                width_sum = 0
                my_face = faces[0] #only get first face,if get all, please set in a loop

                pos_start = tuple([my_face.left(), my_face.top()])
                pos_end = tuple([my_face.right(), my_face.bottom()])
                height = my_face.bottom() - my_face.top()
                width = my_face.right() - my_face.left()
                width_sum += width
                if height > height_max:
                    height_max = height
                else:
                    height_max = height_max

                print("窗口大小："
                      , '\n', "高度 / height:", height_max
                      , '\n', "宽度 / width: ", width_sum)

                img_blank = np.zeros((height_max, width_sum, 3), np.uint8)
                blank_start = 0
                height = my_face.bottom() - my_face.top()
                width = my_face.right() - my_face.left()
                for i in range(height):
                    for j in range(width):
                        img_blank[i][blank_start + j] = img[my_face.top() + i][my_face.left() + j]
                blank_start += width

                file_name = "face_%s.jpg" % time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                cv2.imwrite(os.path.join(save_file_path,file_name), img_blank)
                print("file saved to %s..." % os.path.join(save_file_path,file_name))
                gray_face_path = os.path.join(save_file_path, file_name.split(".")[0]+"new.jpg")
                image_gray = cv2.imread(os.path.join(save_file_path,file_name), cv2.IMREAD_GRAYSCALE)
                #ret, thresh = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
                cv2.imwrite(gray_face_path, image_gray)
                cv2.putText(img_blank, " Face recognized: " + str(len(faces)), (20, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("only 1 face to be saved", img_blank)
                time.sleep(1)

                template_img = Image.open(os.path.join(os.getcwd(), "data", "template.jpg"))
                face_img = Image.open(gray_face_path)
                face_img = face_img.resize((68, 68), Image.ANTIALIAS)
                template_img.paste(face_img, (97, 63))
                template_img.show()


            except Exception as e:
                print("error.. continue capturing: %s" % str(e))
                continue
        else:
            messagebox.showinfo("Sad", "NO face recognized!!")
    if user_input == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
