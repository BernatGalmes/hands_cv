#!/bin/bash

source activate hands-env

echo "Running the script to build the dataset"
cd ModelBuilder
#~/.conda/envs/hands-env/bin/python retrieve_data.py

echo "Generating the results"
cd ../Viewers
~/.conda/envs/hands-env/bin/python probab_maps.py

