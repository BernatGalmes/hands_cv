#!/bin/bash

source activate hands-env

pathImages=""
experimentName="first_iteration"
n_dataset=0
echo "Running the script to build the dataset"
cd ModelBuilder
~/.conda/envs/hands-env/bin/python retrieve_data.py --dataset $n_dataset --n_features 1000 --data_ims_in_file 2000 --data_pixels_class 500

echo "Generating the results"
cd ../Viewers
~/.conda/envs/hands-env/bin/python probab_maps.py --dataset $n_dataset --n_features 1000 --data_ims_in_file 2000 --data_pixels_class 500

