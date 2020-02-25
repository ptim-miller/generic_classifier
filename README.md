# generic_classifier #
CNN - joint research project

## Image Setup ##

This software was created to automatically detect the number of categories to classify, as well as the number of training and validation images. To add images, put image folders under the data_mod folder provided.

Under data_mod/training, add a folder for each category you wish to train your model on. For example, if you wish build an image recognition model for cats, dogs and birds then your folder structure should be:

data_mod/training  
					./birds  
					./cats  
					./dogs  
					
Each training folder should contain an equal number of RGB jpeg images representing each object. 

Validation images should then be places in similar folders under validation. For example:

data_mod/validation  
					./birds  
					./cats  
					./dogs  

Each validation folder should contain an equal number of RGB jpeg images representing each object. 

