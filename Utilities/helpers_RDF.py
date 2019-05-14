import sys
sys.path.insert(0, '../../')

import cv2
import numpy as np
from hands_rdf.hands_rdf.Model.Config import config


def __pack(g, b):
    return (g << 8) | b


v_pack = np.vectorize(__pack)


def read_depth_image(image_path, th_value=config.TH_DEPTH):
    hand_img = cv2.imread(image_path)  # loads as BGR
    hand_mask = hand_img[:, :, 2]  # hand mask is stored in red channel

    # depth is packed in green & blue channels
    depth_image = v_pack(hand_img[:, :, 1], hand_img[:, :, 0]).astype(np.uint16)
    depth_image[(depth_image >= th_value) | (depth_image == 0)] = config.BG_DEPTH_VALUE

    if len(depth_image[depth_image != config.BG_DEPTH_VALUE]) == 0:
        bg_value = config.BG_DEPTH_VALUE
    else:
        bg_value = np.max(depth_image[depth_image != config.BG_DEPTH_VALUE])
        bg_value = bg_value + 200
        depth_image[depth_image == config.BG_DEPTH_VALUE] = bg_value
    return hand_img, hand_mask, depth_image, bg_value


def roi_depth_image(depth_image, bg_value):
    return depth_image < bg_value


def get_prediction(clf, f, depth_image, bg_val, get_proba=True):
    """
    Get the hand prediction probability from a depth image
    :param clf:
    :param f:
    :param depth_image: ndarray
    :param bg_val: integer
    :return:
    """
    indexs = np.nonzero(roi_depth_image(depth_image, bg_val))
    if len(indexs[0]) == 0:
        return np.zeros((depth_image.shape[0], depth_image.shape[1]),
                        dtype=np.float32)
    # Retrieving data
    print("Retrieving data")
    positions, features = f.get_image_features(depth_image, indexs)

    print("predicting")
    proba_mask = np.zeros((depth_image.shape[0], depth_image.shape[1]),
                          dtype=np.float32)
    if get_proba:
        predicted = clf.predict_proba(features)
        proba_mask[positions] = predicted[:, 1]

    else:
        predicted = clf.predict(features)
        proba_mask[positions] = predicted

    return proba_mask