"""
Title:
Get the classification metrics obtained varying most important system params

Description:
The datafiles created are stored in config.FOLDER_RAW_DATA

author:
Bernat Galmés Rubert
"""
print(__doc__)

import logging
from glob import glob

import numpy as np
import pandas as pd

from .Utilities.data_retrieval import retrieve_data

from hands_rdf.hands_rdf.RDF import RDF
from hands_rdf.hands_rdf.Model import config
from hands_rdf.hands_rdf.Model.MultiModels import TestModels

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)
logging.info("PROGRAM START")

offsets_distributions = [
    "NORMAL",
    "UNIFORM",
    "CIRCULARUNIFORM"
]

m_values = [
    75000,
    100000,
    150000,
    200000

]

n_features = np.arange(100, 2000, 200)

depth_thresholds = [
    2000,
    4000,
    8000,
    10000
]

depth_values = [
    4000,
    8000,
    10000,
    np.iinfo(np.uint16).max/2,
    np.iinfo(np.uint16).max
]


dataset = glob(config.PATH_DATASET + "/*")

data = {
        "F1": [],
        "Recall(TPR)": [],
        "Precision": [],
        "Precision_avg": [],
        "Recall_avg": [],
        "F1_avg": [],
        "Support_total": [],
        "cohen_kappa": []
    }
for key in config.__dict__.keys():
    data[key] = []

for dist in offsets_distributions:
    config.OFFSETS_DISTRIBUTION = dist
    for m in m_values:
        config.m = m
        for th_depth in depth_thresholds:
            config.TH_DEPTH = th_depth
            for depth_bg in depth_values:
                config.BG_DEPTH_VALUE = depth_bg
                print("Current config: ")
                print(config.__dict__)
                retrieve_data(dataset, n_pixels_class=750)

                clf = RDF()
                stats = clf.test(TestModels())

                for key in config.__dict__.keys():
                    data[key].append(config.__dict__[key])

                for key in stats.keys():
                    data[key].append(stats[key])

df = pd.DataFrame(data)
excel_writer = pd.ExcelWriter("configurations_" + config.DATASET + ".xlsx", index=False)
df.to_excel(excel_writer, 'features')
excel_writer.save()
