#!/usr/bin/env bash

source activate hands-env

echo "Detect all images faces"
cd ModelBuilder/Faces
#~/.conda/envs/hands-env/bin/python face_detection.py BG_dataset_faces/

echo "Getting the samples of the faces ..."
#~/.conda/envs/hands-env/bin/python face_features_retrieval.py BG_dataset_faces/

echo "Building the dataset ..."
#~/.conda/envs/hands-env/bin/python build_hardneg_set.py BG_dataset_faces/

echo "Generating the results of the built dataset"
cd ../../Viewers
~/.conda/envs/hands-env/bin/python probab_maps.py "fourth_iteration" BG_dataset_faces/ BG_dataset/depth

