import logging
import os
import time
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from hands_cv.Utilities.helpers_RDF import read_depth_image, get_prediction
from hands_rdf.hands_rdf.Model import config
from hands_rdf.hands_rdf.RDF import RDF
from hands_rdf.hands_rdf.features import Features
from ovnImage.functions import check_dir
from ovnImage.plots.InteractivePlot import InteractivePlot


def get_image_prediction(clf, f, image_path):
    print("Evaluating image: ", image_path)
    start_time = time.time()
    im_original, hand_mask, depth_image, bg_val = read_depth_image(image_path)
    print("read: \t --- %s seconds ---" % (time.time() - start_time))

    hand_mask = hand_mask.copy()

    start_time = time.time()
    proba_mask = get_prediction(clf, f, depth_image, bg_val, True)
    print("predict: \t --- %s seconds ---" % (time.time() - start_time))

    _, mask = cv2.threshold(proba_mask, 0.5, 1, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask.astype(np.uint8), 7)
    im2, contours, hierarchy = cv2.findContours(mask,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        x, y, w, h = cv2.boundingRect(cnt)
        im_hand = depth_image[y:y + h, x:x + w]
        depth_min = np.min(im_hand)
        if area < depth_min / 10:
            continue

        cv2.rectangle(im_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print("debugging: ")

    proba_mask[0, 0] = 1.
    images = [
        {
            "img": im_original,
            "title": "Original image"
        },
        {
            "img": depth_image,
            "title": "Depth image",
            "colorbar": True
        },
        {
            "img": hand_mask,
            "title": "Ground truth"
        },
        {
            "img": proba_mask,
            "title": "Probability to be hand",
            "colorbar": True
        }
    ]
    return images


log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)

N_TRY = 2
RES_FOLDER = config.FOLDER_RESULTS + "class_proba/" + str(N_TRY) + "/"
check_dir(RES_FOLDER)

if __name__ == "__main__":
    logging.info("PROGRAM START")

    clf = RDF()
    print(clf)

    # Getting features
    f = Features()
    f.as_DataFrame()

    plotter = InteractivePlot(4)
    plotter.set_authomatic_loop(True, 0.5)

    dataset = glob(config.PATH_DATASET + "/*")
    dataset.sort()
    train, test = train_test_split(dataset, train_size=0.7, random_state=0)
    print(test)

    config.save_txt(RES_FOLDER)
    for i, image_path in enumerate(dataset):
        im_name = os.path.basename(image_path)
        images = get_image_prediction(clf, f, image_path)

        plotter.multi(images, cmap='bwr')
        # plotter.save_multiplot(
        #     RES_FOLDER + im_name,
        #     images,
        #     cmap='bwr'
        # )
