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
keras, tensorflow(-gpu), matplotlib. numpy

#### Expected Output ####
The program creates a results folder using a datetime stamp. A copy of the source code file, the best model found, model_test.py, plot image and a results files are copied to the folder. An example results file is provided based on 100 epochs with a batch size of 128 which was built on greek characters.

## To Test a Resulting Model ##
From within the datetime results folder desired (ie results/20200226-012834), run the test program with:

##### python3 model_test.py beta.jpg #####

Test images are provided in the results->images folder. The model_test program automatically looks to this folder when run, 

#### Python Packages ####
numpy, PIL, keras, tensorflow, warnings, os, sys

#### Expected Output ####

user@GPU:~/git/generic_classifier/results/20200226-012834$ python3 model_test.py beta.jpg  
Using TensorFlow backend.  
Total Possible : 24  
Beta : 99.91%  

user@GPU:~/git/generic_classifier/results/20200226-012834$ python3 model_test.py omega.jpg  
Using TensorFlow backend.  
Total Possible : 24  
Omega : 99.99%  

user@GPU:~/git/generic_classifier/results/20200226-012834$ python3 model_test.py psi.jpg  
Using TensorFlow backend.  
Total Possible : 24  
Psi : 99.37%  

user@GPU:~/git/generic_classifier/results/20200226-012834$ python3 model_test.py xi.jpg  
Using TensorFlow backend.  
Total Possible : 24  
Xi : 99.4%  

## Modify Parameters ##
In train.py, alter these variables if desired:  
* epochs = 100
* batch_size = 128  

Other items can be changed to alter outcome of model. See comments throughout file. For example, 
image augmentation can be accomplished by changing attributes in train.py:

train_datagen=ImageDataGenerator(
    rescale=1./255,
    zoom_range=.4,
    shear_range=0.0,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    vertical_flip=False,
    horizontal_flip=False
)

Modifying these items to higher ranges will alter the images being trained on randomly. The current 
settings are good for character training, but bad for general images of objects like dogs and cats.  



