## Author: P. Tim Miller
## (c) P. Tim Miller / John Wallin
## All Rights Reserved
## Academic use only
## MTSU Research
## Computational Science

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from keras.preprocessing import image
    from keras.models import load_model
    import sys
    from PIL import Image
    import numpy as np
    from PIL import Image, ImageOps, ImageEnhance
    import PIL.ImageOps
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import h5py

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

## Get categories from categories.txt
with open("categories.txt") as f:
    LABELS_LIST = [line.rstrip() for line in f]
num_classes = len(LABELS_LIST)
print("Total Possible :", num_classes)
print("All Categories :", LABELS_LIST)

test_image = PIL.Image

try:
    file = sys.argv[1]
    in_image = Image.open('../images/' + file)
    in_image = in_image.resize((70, 70), PIL.Image.ANTIALIAS)
    in_image = in_image.convert("RGB") ## convert this to "L" for a model trained on black and white images
except Exception as e:
    print("*** File read error - Usage: python3 model_test.py image.jpg")
    print(e)
    exit()

model = load_model('model.hdf5')
test_image = image.img_to_array(in_image)
test_image = (test_image/255.)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
y_prob = model.predict_classes(test_image)
y_classes = result.argmax(axis=-1)
print(LABELS_LIST[int(y_prob)] + ' : ' + str(round(result[0][int(y_prob)] * 100, 2)) + '%')
