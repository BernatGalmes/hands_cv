"""
Title:
Create the data files from the dataset images.

Description:
The datafiles created are stored in config.FOLDER_RAW_DATA

author:
Bernat Galmés Rubert
"""
print(__doc__)

import sys
sys.path.insert(0, '../../')

import argparse
import logging
from glob import glob

from hands_rdf.hands_rdf.Model import config
from hands_cv.Utilities.data_retrieval import retrieve_data


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--path_images', help='foo help')

args = config.set_arguments(parser)
if args.path_images:
    PATH_IMAGES = config.PATH_DATASETS + args.path_images
else:
    PATH_IMAGES = config.PATH_DATASET

if __name__ == "__main__":

    dataset = glob(PATH_IMAGES + "/*")
    dataset.sort()

    retrieve_data(dataset)
