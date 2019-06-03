"""
Title:
Create data files with a specified percentage of hard negatives

Description:
Create n data files with a fixed number of samples belonging to the face

author:
Bernat Galmés Rubert
"""
print(__doc__)
import sys
sys.path.insert(0, '../../../')

import logging
import argparse
import numpy as np

from glob import glob

from hands_rdf.hands_rdf.Model.MultiModels import TrainModels, TestModels
from hands_rdf.hands_rdf.Model.Config import config
from ovnImage import check_dir

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

logging.info("Running version: " + str(config.VERSION))

parser = argparse.ArgumentParser()
parser.add_argument('--path_dst', help='foo help')
parser.add_argument('--no_faces', help='foo help',
                    action="store_true")

args = config.set_arguments(parser)
if args.path_dst:
    PATH_DATA = config.FOLDER_RAW + args.path_dst

else:
    PATH_DATA = config.FOLDER_RAW_DATA + "faces_data/"

TAKE_FACES_DATA = not args.no_faces

FOLDER_train = PATH_DATA + "train/"
FOLDER_test = PATH_DATA + "test/"

check_dir(FOLDER_train)
check_dir(FOLDER_test)


# Specify the list of datasets to use to train
DATASETS_train = [
    "BG_dataset/depth"
]

# Specify the list of datasets to use to test the classifier
DATASETS_test = [
    "BG_dataset/depth"
]


def build_set(models, paths_faces_models, path_save, take_faces=True):
    """

    :param models: MultiModels subsets of data files
    :param paths_faces_models: list of *.npy files with the faces data
    :param path_save: String Folder path where save the resuling matrix

    :return: None
    """
    n_models = len(models)

    models_indexs = []


    logging.info("selecting indexes ...")

    # mark model indexes
    for i, (file, model) in enumerate(models):
        indexs = np.arange(0, model.shape[0], step=1)
        np.random.shuffle(indexs)
        models_indexs.append(
            np.array_split(indexs, n_models)
        )
    models_indexs = np.array(models_indexs)

    if take_faces:
        faces_models_indexs = []
        for i, path in enumerate(paths_faces_models):
            faces_data = np.load(path)
            indexs = np.arange(0, faces_data.shape[0], step=1)
            np.random.shuffle(indexs)
            faces_models_indexs.append(
                np.array_split(indexs, n_models)
            )
        faces_models_indexs = np.array(faces_models_indexs)

    logging.info("\ncreating models ...")
    for n_file in range(n_models):
        logging.info("Building matrix ...")

        n_samples_file = 0
        for i in range(len(models)):
            n_samples_file += len(models_indexs[i][n_file])
        if take_faces:
            for i in range(len(paths_faces_models)):
                n_samples_file += len(faces_models_indexs[i][n_file])

        logging.debug("number of samples in file:" + str(n_samples_file))
        idx_row = 0
        samples = np.zeros((n_samples_file, models[0][1].shape[1]), dtype=np.int8)
        logging.debug("shape: " + str(samples.shape))
        for i, (file, model) in enumerate(models):
            model_samples = model[models_indexs[i][n_file], :]
            samples[idx_row:idx_row + len(model_samples), :] = model_samples
            idx_row = idx_row + len(model_samples)
        if take_faces:
            for i, path in enumerate(paths_faces_models):
                faces_data = np.load(path)

                face_samples = faces_data[faces_models_indexs[i][n_file], :]
                samples[idx_row:idx_row + len(face_samples), :] = face_samples
                idx_row = idx_row + len(face_samples)

        file_name = path_save + "data_" + str(n_file) + ".npy"
        save_matrix(samples, file_name)


def save_matrix(matrix, file_name):
    np.save(file_name, matrix)


paths_faces_models = glob(config.FOLDER_RAW_DATA + "faces_data_2/*.npy")
for ds in DATASETS_train:
    config.DATASET = ds
    models = TrainModels(mode='npy')
    build_set(models, paths_faces_models[:int(len(paths_faces_models)*0.7)], FOLDER_train, take_faces=TAKE_FACES_DATA)


for ds in DATASETS_test:
    config.DATASET = ds
    models = TestModels(mode='npy')
    build_set(models, paths_faces_models[int(len(paths_faces_models)*0.7):], FOLDER_test, take_faces=TAKE_FACES_DATA)
