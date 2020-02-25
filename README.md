# Generic Classifier #
CNN - joint research project

## Image Setup ##
This software was created to automatically detect the number of categories to classify, as well as the number of training and validation images. To add images, put image folders under the data_mod folder provided.

Under data_mod/train, add a folder for each category you wish to train your model on. For example, if you wish build an image recognition model for cats, dogs and birds then your folder structure should be:

data_mod/train  
					./birds  
					./cats  
					./dogs  
					
Each training folder should contain an equal number of RGB jpeg images representing each object. 

Validation images should then be places in similar folders under data_mod/validate. For example:

data_mod/validate  
					./birds  
					./cats  
					./dogs  

Each validation folder should contain an equal number of RGB jpeg images representing each object. 

## To Run ##
From top level directory, run the program with:

##### python3 train.py #####

#### Python Packages ####
shutil, keras, tensorflow(gpu), matplotlib. numpy, time, timeit, warnings, os

#### Expected Output ####
The program creates a results folder using a datetime stamp. A copy of the source code file, the best model found, model_test.py, plot image and a results files are copied to the folder. An example results file is provided based on 200 epochs with a batch size of 50 which was built on greek characters.

## To Test a Resulting Model ##
From within the datetime results folder desired (ie results/20200224-214702), run the test program with:

##### python3 model_test.py beta.jpg #####

Test images are provided in the results->images folder. The model_test program automatically looks to this folder when run, The beta.jpg file is a valid greek character image and omega.jpg is a generic black and white image.

#### Python Packages ####
numpy, PIL, keras, tensorflow, warnings, os, sys

#### Expected Output ####

user@GPU:~/git/generic_classifier/results/20200224-214702$ python3 model_test.py beta.jpg  
Using TensorFlow backend.  
Total Possible : 24  
Beta : 99.93%  
  
user@GPU:~/git/generic_classifier/results/20200224-214702$ python3 model_test.py omega.jpg  
Using TensorFlow backend.  
Total Possible : 24  
Omega : 36.4%  

## Modify Parameters ##
In train.py, alter these variables if desired:  
* epochs = 200  
* batch_size = 50  

Other items can be changed to alter outcome of model. See comments throughout file.  

