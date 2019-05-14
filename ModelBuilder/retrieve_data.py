"""
Title:
Create the data files from the dataset images.

Description:
The datafiles created are stored in config.FOLDER_RAW_DATA

author:
Bernat Galmés Rubert
"""
print(__doc__)
import logging
from glob import glob

from hands_rdf.hands_rdf.Model import config
from hands_cv.Utilities.data_retrieval import retrieve_data

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    dataset = glob(config.PATH_DATASET + "/*")
    print(config.PATH_DATASET + "/*")
    dataset.sort()
    # Split the dataset in two parts:
    retrieve_data(dataset)
