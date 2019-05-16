"""
Title:
Compute face positions from a frames set

Description:
Compute and save in a file all faces postitions

Sources:
https://www.superdatascience.com/opencv-face-detection/
https://github.com/mpatacchiola/deepgaze

author:
Bernat Galmés Rubert
"""
import os
import sys
sys.path.insert(0, '../../../')

import cv2

from glob import glob
import numpy as np
import pandas as pd

from hands_rdf.hands_rdf.Model.Config import config
from hands_rdf.hands_rdf.features import Features

from ovnImage import InteractivePlot, check_dir, images2video

from deepgaze.deepgaze.face_detection import HaarFaceDetector

PATH_DATASET = config.PATH_DATASETS + "/frames/BG/"

PATH_COLOR = PATH_DATASET + "color/"
PATH_DEPTH = PATH_DATASET + "depth/"

PATH_RESULTS = config.FOLDER_RESULTS + "faces/"
check_dir(PATH_RESULTS)


if len(sys.argv) > 1:
    PATH_DATA = config.FOLDER_RAW + sys.argv[1]


else:
    PATH_DATA = config.FOLDER_RAW + "faces_data/"

check_dir(PATH_DATA)
PATH_FACES_DATA = PATH_DATA + "/faces_squares.xlsx"


def hardcore_face_detection(im_gray):

    faces = haar_face_cascade.detectMultiScale(im_gray,
                                               scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("using lbf ...")
        faces = lbp_face_cascade.detectMultiScale(im_gray,
                                                  scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print("using deepgaze implementation ...")
            faces = hfd.returnMultipleFacesPosition(im_gray, runFrontal=True, runFrontalRotated=True,
                                                    runLeft=True, runRight=True,
                                                    frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2,
                                                    leftScaleFactor=1.15, rightScaleFactor=1.15,
                                                    minSizeX=64, minSizeY=64,
                                                    rotationAngleCCW=30, rotationAngleCW=-30)

    return faces


if __name__ == "__main__":

    # plotter = InteractivePlot(4)
    # plotter.set_authomatic_loop(True, 0.5)

    dataset = glob(PATH_COLOR + "/*")
    dataset.sort()

    f = Features(True)
    haar_face_cascade = cv2.CascadeClassifier(config.FOLDER_DATA + '/face_detection/haarcascade_frontalface_alt.xml')
    lbp_face_cascade = cv2.CascadeClassifier(config.FOLDER_DATA + '/face_detection/lbpcascade_frontalface.xml')
    hfd = HaarFaceDetector("../../../deepgaze/etc/xml/haarcascade_frontalface_alt.xml",
                           "../../../deepgaze/etc/xml/haarcascade_profileface.xml")
    faces_data = {
        "image": [],
        "x": [],
        "y": [],
        "w": [],
        "h": []
    }
    for i, image_path in enumerate(dataset[100:]):
        im_name = os.path.basename(image_path)
        im = cv2.imread(image_path, -1)

        im_debug = im.copy()
        im_debug = cv2.cvtColor(im_debug, cv2.COLOR_BGRA2RGB)

        foreground = im[:, :, 3]
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
        im[im == [0, 0, 0]] = 255

        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        faces = hardcore_face_detection(im_gray)
        n_faces = len(faces)

        if n_faces != 0:

            for (x, y, w, h) in faces:
                head_foreground = foreground[y:y+h, x:x+w]

                if np.count_nonzero(head_foreground) > (h*w)*0.9:
                    faces_data["image"].append(im_name)
                    faces_data["x"].append(x)
                    faces_data["y"].append(y)
                    faces_data["w"].append(w)
                    faces_data["h"].append(h)

                    cv2.rectangle(im_debug, (x, y), (x+w, y+h), (0, 255, 0), 1)
                else:
                    print("Is not onto the face")

        # print the number of faces found
        print(i, '.-. Faces found:', n_faces)

        cv2.imwrite(PATH_RESULTS + "im_" + str(i) + ".png", im_debug)
        # plotter.multi([{
        #     "img": im,
        #     "title": "im"
        # },{
        #     "img": im_debug,
        #     "title": "Face squares"
        # }], cmap='bwr')

    images2video(sorted(glob(PATH_RESULTS + "*.png")),
                 PATH_RESULTS + "video.avi", 4)

    faces_df = pd.DataFrame(faces_data)
    excel_writer = pd.ExcelWriter(PATH_FACES_DATA, index=False)
    faces_df.to_excel(excel_writer, sheet_name='data', index=False)
    excel_writer.save()
