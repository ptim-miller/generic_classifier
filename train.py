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
    import timeit
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import keras
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from keras.models import Sequential
    from keras.preprocessing.image import ImageDataGenerator
    from keras.layers import Dense, Flatten, Dropout, BatchNormalization
    from keras.layers import Conv2D
    from keras import regularizers
    import shutil

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def file_count(path):
    file_count = 0
    folder_count = 0
    for subdir, dirs, files in os.walk(path):
        folder_count += 1
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                file_count += 1
    return file_count, folder_count - 1

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

## set path to train and val
train_dir = 'data_mod/train/'
validation_dir = 'data_mod/validate/'

## determine num of categories
categories  = os.listdir(train_dir)
num_classes = len(categories)
categories.sort()
categories_val  = os.listdir(validation_dir)
num_classes_val = len(categories_val)

## validate matching count train and val
if(num_classes != num_classes_val):
    print("Training category count does not match Validation category count.")
    exit()

## get image counts
train_samples = file_count(train_dir)
validation_samples = file_count(validation_dir)

# TODO - override with command line args
img_width, img_height = 70, 70
epochs = 200
batch_size = 50
depth=3
color_mode='rgb'

## uncomment to use bw to train, be sure to change model_test.py
## change convert("RGB") to convert("L") if model is built on bw
# bw=False
# if (bw):
#    depth=1
#    color_mode='grayscale'

if K.image_data_format() == 'channels_first':
    input_shape = (depth, img_width, img_height)
else:
    input_shape = (img_width, img_height, depth)

def cnn():
    weight_decay = 1e-4
    model = Sequential()

    model.add(Conv2D(16, (1, 1), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (2, 2), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1), dilation_rate=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (1, 1), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (2, 2), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (1, 1), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2, 2), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = cnn()

opt = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

model.summary()

start = timeit.default_timer()

timestr = time.strftime("%Y%m%d-%H%M%S")
save_folder = './results/' + timestr
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
try:
    shutil.copy('./train.py', save_folder + '/train.py')
except:
    print("Unable to copy source file.")
fileName = save_folder + '/model.hdf5'

early_stopping = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

checkpoint = [ModelCheckpoint(
    filepath=fileName,
    monitor='val_categorical_accuracy',
    verbose=1, save_best_only=True),
    # early_stopping, ## uncomment if early stop desired
    LearningRateScheduler(lr_schedule)]

train_datagen=ImageDataGenerator(
    rescale=1./255,
    zoom_range=.7,
    shear_range=0.05,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    vertical_flip=False,
    horizontal_flip=False
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode=color_mode,
    seed=42)

## Uncomment to review augmented examples
## dont use rescale=1./255 if viewing
# for i in range(9):
#     plt.subplot(330 + 1 + i)
#     batch = train_generator.next()
#     image = batch[0].astype('uint8')
#     plt.imshow(image[0])
# plt.show()
# label_map = (train_generator.class_indices)
# print(label_map)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode=color_mode)

step_train=train_generator.n//train_generator.batch_size
step_valid=validation_generator.n//validation_generator.batch_size

history = model.fit_generator(
    train_generator,
    steps_per_epoch=step_train,
    epochs=epochs,
    callbacks=checkpoint,
    validation_data=validation_generator,
    validation_steps=step_valid)

stop = timeit.default_timer()
minutes = (stop - start)/60.
runtime = ('Time: ' + str(round(minutes, 2)))
acc_results = np.asarray(history.history['val_categorical_accuracy'])
loss_results = np.asarray(history.history['val_loss'])
best = acc_results.argmax()
loss = loss_results[best]
acc = acc_results[best]

# print results to screen
print('Test loss:', loss )
print('Test accuracy:', acc )
print('Runtime: ' + runtime + ' minutes')

# write results to file
f = open(save_folder +'/results.txt', 'w')
f.write('\n' + 'File: ' + str(fileName) +'\n')
f.write('Test Loss     : ' + str(loss) +'\n')
f.write('Test Accuracy : ' + str(acc) +'\n')
f.write('Runtime: ' + runtime + ' minutes\n\n')
f.close()

# plot results
fig = plt.figure(1)
plt.subplot(211)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

## uncomment next line to show plot
# plt.show()

## save plot to save folder along with model and results
fig.savefig(save_folder + '/plot.png')

test_file = '/model_test.py'

try:
    with open('./' + save_folder + '/categories.txt', 'w') as f:
        for item in categories:
            f.write("%s\n" % item)
except:
    print("Unable to create categories file.")

try:
    shutil.copy('./' + test_file , save_folder + test_file)
except:
    print("Unable to copy test file.")
