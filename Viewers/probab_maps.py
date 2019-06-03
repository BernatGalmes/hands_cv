import sys
sys.path.insert(0, '../../')

import argparse
import logging
import os
import time
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from hands_cv.Utilities.helpers_RDF import read_depth_image, get_prediction
from sklearn.metrics import cohen_kappa_score, classification_report
from hands_rdf.hands_rdf.Model import config
from hands_rdf.hands_rdf.RDF import RDF
from hands_rdf.hands_rdf.features import Features
from hands_rdf.hands_rdf.helpers import show_stats

from ovnImage.functions import check_dir
from ovnImage.plots.InteractivePlot import InteractivePlot, MultiPlot
from ovnImage import images2video


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

do_video = True
N_TRY = 1

parser = argparse.ArgumentParser()

parser.add_argument('--name', help='foo help')
parser.add_argument('--path_dataset', help='foo help')
parser.add_argument('--path_images', help='foo help')

args = config.set_arguments(parser)
if args.path_images:
    PATH_IMAGES = config.PATH_DATASETS + args.path_images
else:
    PATH_IMAGES = config.PATH_DATASET

if args.path_dataset:
    config.DATASET = args.path_dataset
    
if args.name:
    experiment_name = args.name

else:
    experiment_name = "proba_images"

folder_results = config.FOLDER_RESULTS + experiment_name + "/" + str(N_TRY) + "/"
folder_results_images = folder_results + "images/"

video_fps = 1

check_dir(folder_results_images)

config.set_arguments()

if __name__ == "__main__":
    logging.info("PROGRAM START")

    logging.info("Building RDF using files in " + config.FOLDER_RAW_DATA)
    clf = RDF()
    from hands_rdf.hands_rdf.Model.MultiModels import TestModels

    y_real_all = []
    y_pred_all = []

    for file, model in TestModels(mode="npy"):
        X_test, y_test = model[:, :-1], model[:, -1]
        y_pred = clf.predict(X_test)

        y_real_all.extend(y_test)
        y_pred_all.extend(y_pred)

    results = show_stats(y_real_all, y_pred_all)

    with open(folder_results + "report.txt", "w") as fw:
        fw.write(classification_report(y_real_all, y_pred_all))

        cohenKappa = cohen_kappa_score(y_real_all, y_pred_all)
        fw.write("Cohen kappa score: " + str(cohenKappa))

    print(clf)

    # Getting features
    f = Features()
    f.as_DataFrame()

    if do_video:
        plotter = MultiPlot(4)
    else:
        plotter = InteractivePlot(4)
        plotter.set_authomatic_loop(True, 0.5)

    dataset = glob(PATH_IMAGES + "/*")
    dataset.sort()

    train, test = train_test_split(dataset, train_size=0.7, random_state=0)

    config.save_txt(folder_results)
    for i, image_path in enumerate(test):
        im_name = os.path.basename(image_path).split(".")[0]
        images = get_image_prediction(clf, f, image_path)

        if do_video:
            plotter.save_multiplot(
                folder_results_images + im_name + ".png",
                images,
                cmap='bwr'
            )

        else:
            plotter.multi(images, cmap='bwr')

    if do_video:
        images2video(sorted(glob(folder_results_images + "*.png")),
                     folder_results + experiment_name + ".avi", video_fps)
