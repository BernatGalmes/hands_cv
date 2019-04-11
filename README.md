# hands_cv

Executable files to test the hands segmentation method from hands_rdf
library (https://github.com/BernatGalmes/hands_rdf).


## Executable files
This repository contains two main executable files. One to build
the data model need to train the RDF of the target library, from the images
in the data set.

The other executable is to build and test the RDF.


### retrieve_data.py
This executable file build the data model need for the hands_rdf library to
build the classifier. It builds both train and test set.

It is build from the images that contain the dataset in the folder
specified in the config file of the hands_rdf library.


### probab_maps.py
The function of this executable is train and show the results of the
RDF.

The execution of this script train a RDF classifier with the data build
with retrieve_data.py script. Once it is trained it shows a set of metrics
obtained predicting the test set ann then build a debug images from all
the images in the data set that shows the results of the predictions visually.


