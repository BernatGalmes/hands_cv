"""
Face Detection example

Using basic opencv classificators and a wrap class

https://www.superdatascience.com/opencv-face-detection/
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

from ovnImage import InteractivePlot, check_dir
from scipy.spatial import distance

FACES_CLASS = 2


PATH_DATASET = config.PATH_DATASETS + "/frames/BG/"

PATH_COLOR = PATH_DATASET + "color/"
PATH_DEPTH = PATH_DATASET + "depth/"


if len(sys.argv) > 1:
    PATH_DATA = config.FOLDER_RAW + sys.argv[1]

else:
    PATH_DATA = config.FOLDER_RAW_DATA + "faces_data"


FILE_FACES = PATH_DATA + "/faces_squares.xlsx"

PATH_DATASET = PATH_DATA + "/data/"
check_dir(PATH_DATASET)


def process_image(f, image_path, data, faces, idx_sample, n_face_samples):
    im_name = os.path.basename(image_path)
    im = cv2.imread(image_path, -1)
    foreground = im[:, :, 3]
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    im[im == [0, 0, 0]] = 255

    im_depth = cv2.imread(PATH_DEPTH + im_name, cv2.IMREAD_ANYDEPTH)
    im_depth[~foreground.astype(np.bool)] = config.BG_DEPTH_VALUE
    im_depth[(im_depth >= config.TH_DEPTH) | (im_depth == 0)] = config.BG_DEPTH_VALUE

    if len(im_depth[im_depth != config.BG_DEPTH_VALUE]) == 0:
        bg_value = config.BG_DEPTH_VALUE

    else:
        bg_value = np.max(im_depth[im_depth != config.BG_DEPTH_VALUE])
        bg_value = bg_value + 200
        im_depth[im_depth == config.BG_DEPTH_VALUE] = bg_value

    for j, face in faces.iterrows():
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        face_center = np.array([y + (h / 2), x + (w / 2)])
        face_radius = int(distance.euclidean(face_center, (y, x)))
        head_foreground = foreground[y:y + h, x:x + w]
        if np.count_nonzero(head_foreground) > (h * w) * 0.9:

            # build head mask
            head_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
            cv2.circle(head_mask, (int(face_center[1]), int(face_center[0])), face_radius, 1, -1)
            head_mask = head_mask & foreground

            # seleccionar 100 pixels de cara aleatories de cada imatge
            y_head, x_head = head_mask.nonzero()
            if len(y_head) > n_face_samples:
                idx = np.random.uniform(0, len(y_head), n_face_samples).astype(np.uint16)
                y_head, x_head = (y_head[idx], x_head[idx])
            else:
                continue

            positions, feats = f.get_image_features(im_depth, (y_head, x_head))
            if idx_sample + (len(positions[0])) < N_FILE_SAMPLES:
                n_write_samples = len(positions[0])
            else:
                n_write_samples = N_FILE_SAMPLES - idx_sample

            for i in range(n_write_samples):
                data[i + idx_sample, :-1] = feats[i]
                data[i + idx_sample, -1] = FACES_CLASS  # is Hand?
            idx_sample += n_write_samples

        else:
            print("Is not onto the face")

    return idx_sample


def generate_faces_data(dataset, df_faces, n_face_samples, n_samples_file):
    f = Features(all_offsets=True)
    idx_image = 0
    i_file = 0
    while idx_image < len(dataset):
        idx_sample = 0
        data = np.zeros((n_samples_file, config.N_FEATURES + 1), dtype=np.int8)
        while idx_sample < n_samples_file:
            image_path = dataset[idx_image]
            im_name = os.path.basename(image_path)

            faces = df_faces[df_faces.loc[:, 'image'] == im_name]
            if not faces.empty:
                idx_sample = process_image(f, image_path, data, faces, idx_sample, n_face_samples)
            idx_image += 1
            if idx_image >= len(dataset):
                # cut matrix and exit
                data = np.delete(data, np.s_[idx_sample:], 0)
                break
        i_file += 1
        print("saving file", PATH_DATASET + "data_" + str(i_file) + ".npy")
        np.save(PATH_DATASET + "data_" + str(i_file) + ".npy", data)


N_SAMPLES_FACE = 500
N_FILE_SAMPLES = 1000000

if __name__ == "__main__":

    dataset = glob(PATH_COLOR + "/*")
    dataset.sort()

    df_faces = pd.read_excel(FILE_FACES, sheet_name="data")

    generate_faces_data(dataset, df_faces, N_SAMPLES_FACE, N_FILE_SAMPLES)
