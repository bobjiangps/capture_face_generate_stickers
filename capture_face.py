import dlib
import numpy as np
import cv2
import time
import os
import platform
import configparser
from tkinter import messagebox
from PIL import Image


def load_config():
    config_file = os.path.join(os.path.join(os.getcwd(), "config", "run_config.ini"))
    cf = configparser.ConfigParser()
    cf.read(config_file)
    return dict(cf.items("Run"))


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_face_area(face):
    area_height = 0
    area_width = 0
    height = face.bottom() - face.top()
    width = face.right() - face.left()
    area_width += width
    if height > area_height:
        area_height = height
    print("face area", "height:", area_height, "width: ", area_width)
    return area_height, area_width


if __name__ == "__main__":
    config = load_config()

    save_file_path = os.path.join(os.getcwd(), "saved_face")
    create_dir(save_file_path)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    print(cap.get(3), cap.get(4))
    data_path = os.path.join(os.getcwd(), "data")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(os.getcwd(), "data", "shape_predictor_68_face_landmarks.dat"))

    while cap.isOpened():
        ret, img = cap.read()
        cv2.imshow("show emotion on your face, press g to generate sticker", img)
        user_input = cv2.waitKey(1)
        if user_input == ord('g'):
            img_from_camera = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = detector(img_from_camera, 1)
            print("there are %d faces" % len(faces))
            font = cv2.FONT_HERSHEY_SIMPLEX
            if len(faces) > 0:
                try:
                    my_face = faces[0]  # only get first face
                    area_height, area_width = get_face_area(my_face)
                    img_blank = np.zeros((area_height, area_width, 3), np.uint8)
                    blank_start = 0
                    height = my_face.bottom() - my_face.top()
                    width = my_face.right() - my_face.left()
                    for i in range(height):
                        for j in range(width):
                            img_blank[i][blank_start + j] = img[my_face.top() + i][my_face.left() + j]

                    file_name = "face_%s.jpg" % time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                    cv2.imwrite(os.path.join(save_file_path, file_name), img_blank)
                    print("file saved to %s..." % os.path.join(save_file_path, file_name))
                    gray_face_path = os.path.join(save_file_path, file_name.split(".")[0] + "new.jpg")

                    # show the gray image
                    # image_gray = cv2.imread(os.path.join(save_file_path,file_name), cv2.IMREAD_GRAYSCALE)
                    # cv2.imwrite(gray_face_path, image_gray)

                    # show the image of Thresholding of gray image, kind 1
                    # image_gray = cv2.imread(os.path.join(save_file_path,file_name), cv2.IMREAD_GRAYSCALE)
                    # ret, thresh = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
                    # cv2.imwrite(gray_face_path, thresh)

                    # show the image of Thresholding of gray image, kind 2
                    image_gray = cv2.cvtColor(cv2.imread(os.path.join(save_file_path, file_name)), cv2.COLOR_RGB2GRAY)
                    # binary = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
                    # binary = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
                    binary = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)
                    cv2.imwrite(gray_face_path, binary)

                    cv2.putText(img_blank, str(len(faces)) + " Face recognized ", (20, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    shape = predictor(img, my_face)
                    for pt in shape.parts():
                        pt_pos = (pt.x, pt.y)
                        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
                    cv2.imshow("feature points", img)
                    print("feature points coordinates: %s" % str(shape.parts()))

                    cv2.imshow("only 1 face to be saved", img_blank)
                    time.sleep(1)

                    template_img = Image.open(os.path.join(os.getcwd(), "data", config["template_pic"]))
                    face_img = Image.open(gray_face_path)
                    face_img = face_img.resize((68, 68), Image.ANTIALIAS)
                    template_img.paste(face_img, (int(config["pic_paste_x"]), int(config["pic_paste_y"])))
                    template_img.show()
                    print("generate stickers completely")
                except Exception as e:
                    print("error.. continue capturing: %s" % str(e))
                    continue
            else:
                if platform.system().find("Darwin") < 0:
                    messagebox.showinfo("Sad", "NO face recognized!!")
                else:
                    img_message = np.zeros((200, 200, 3), np.uint8)
                    cv2.putText(img_message, str(len(faces)) + " Face recognized ", (20, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("NO face recognized!!", img_message)

        if user_input == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
