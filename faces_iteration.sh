#!/usr/bin/env bash

source activate hands-env

echo "Detect all images faces"
cd ModelBuilder/Faces
~/.conda/envs/hands-env/bin/python face_detection.py BG_dataset_faces/

echo "Getting the samples of the faces ..."
~/.conda/envs/hands-env/bin/python face_features_retrieval.py BG_dataset_faces/

echo "Building the dataset ..."
~/.conda/envs/hands-env/bin/python build_hardneg_set.py --path_dst BG_dataset_faces/

echo "Generating the results of the built dataset"
cd ../../Viewers
~/.conda/envs/hands-env/bin/python probab_maps.py --name "faces_data" --n_train_set 2 --n_test_set 0 --path_images BG_dataset/depth
