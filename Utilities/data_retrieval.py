import os
import json
import logging
import threading
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .helpers_RDF import read_depth_image, roi_depth_image
from hands_rdf.hands_rdf.features import Features
from hands_rdf.hands_rdf.Model import config
from ovnImage import check_dir

PIXELS_FOR_CLASS = config.DATA_PIXELS_CLASS
IMAGES_IN_FILE = config.DATA_IMS_IN_FILE


def __save_data(data, i_group, save_path):

    file_name = save_path + "data_" + str(i_group) + ".npy"
    np.save(file_name, data)

    logging.debug('Writted: ' + file_name)


def __save_info_files(f, save_path):
    excel_writer = pd.ExcelWriter(save_path + "features.xlsx", index=False)
    f.as_DataFrame().to_excel(excel_writer, 'features')
    config.save_txt(save_path)
    with open(save_path + "stats.json", 'w') as fp:
        json.dump(f.get_metaData(), fp, sort_keys=True, indent=4)
    pd.DataFrame().to_excel(excel_writer, 'metadata')
    excel_writer.save()


def build_data(dataset: list, save_path: str, n_pixels_class: int=PIXELS_FOR_CLASS):
    """

    :param dataset: list of images
    :param save_path: string with the path where we store the data files build
    :param n_pixels_class:
    :return:
    """
    if os.path.exists(save_path) and os.path.isdir(save_path):
        shutil.rmtree(save_path)
    n_samples_img = 2*n_pixels_class

    check_dir(save_path)
    f = Features(all_offsets=True)

    # Save information files
    __save_info_files(f, save_path)

    # Create files from a group of images in the dataset
    ini_group = 0
    last_im_group = 0
    i_group = 0
    while last_im_group != len(dataset):
        last_im_group = ini_group + IMAGES_IN_FILE

        if last_im_group > len(dataset):
            last_im_group = len(dataset)
        n_images_group = last_im_group - ini_group
        write_pos = 0
        data = np.zeros((n_images_group*n_samples_img, config.N_FEATURES+1), dtype=np.int8)
        images = dataset[ini_group:last_im_group]
        ini_group = last_im_group
        for idx, image_path in enumerate(images):
            try:
                _, hand_mask, depth_image, bg_value = read_depth_image(image_path)
            except:
                logging.error('Erronious image:' + image_path)
                continue

            body_mask = (hand_mask == 0) & roi_depth_image(depth_image, bg_value)
            y_hand, x_hand = np.nonzero(hand_mask.astype(np.bool) & roi_depth_image(depth_image, bg_value))
            y_body, x_body = np.nonzero(body_mask)
            if len(y_body) == 0 or len(y_hand) == 0:
                logging.debug('Skipping: Image with no body or hand pixels')
                continue

            n_pixels = np.min([n_pixels_class, len(y_body)])
            i = np.random.uniform(0, len(y_body), n_pixels).astype(np.uint16)
            i_body = (y_body[i], x_body[i])

            n_pixels = np.min([n_pixels_class, len(y_hand)])
            i = np.random.uniform(0, len(y_hand), n_pixels).astype(np.uint16)
            i_Hand = (y_hand[i], x_hand[i])

            indexs = tuple(np.hstack((i_Hand, i_body)))

            positions, feats = f.get_image_features(depth_image, indexs)
            for i in range(len(positions[0])):
                y, x = positions[0][i], positions[1][i]

                data[write_pos, :-1] = feats[i]
                data[write_pos, -1] = hand_mask[y][x] != 0
                write_pos += 1

        if write_pos < n_images_group*n_samples_img:
            logging.info("Cutting matrix to " + str(write_pos) + "position")
            # cut matrix and exit
            data = np.delete(data, np.s_[write_pos:], 0)

        i_group += 1
        t1 = threading.Thread(name='block',
                              target=__save_data,
                              args=(data, i_group, save_path))
        t1.start()

    logging.debug('Waiting for writting threads')
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()
    logging.debug('Processes ended')


def retrieve_data(dataset, n_pixels_class=PIXELS_FOR_CLASS):
    print("total images: %", len(dataset))
    train_size = 0.7

    train, test = train_test_split(dataset, train_size=train_size, random_state=0)
    build_data(train, config.FOLDER_TRAIN, n_pixels_class=n_pixels_class)
    build_data(test, config.FOLDER_TEST)
