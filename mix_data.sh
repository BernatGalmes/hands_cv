#!/usr/bin/env bash

source activate hands-env

n_dataset=0
echo "Running the script to build the dataset"
cd ModelBuilder
#~/.conda/envs/hands-env/bin/python retrieve_data.py --n_dataset ${n_dataset} --n_features 1000 --data_ims_in_file 2000 --data_pixels_class 500

echo "Building the dataset ..."
cd Faces
~/.conda/envs/hands-env/bin/python build_hardneg_set.py --path_dst mix_data_nyu/ --n_dataset ${n_dataset} --no_faces

echo "Generating the results of the built dataset"
cd ../../Viewers
~/.conda/envs/hands-env/bin/python probab_maps.py --name "mix_data_nyu" --n_train_set 4 --n_test_set 1 --path_images BG_dataset/depth