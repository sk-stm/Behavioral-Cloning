import csv
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

# DATA hyper parameters
correction = 0.1
ADD_AUGMENTED = True
ADD_LR = False

# read the csv file
samples = []
# I only train on my laptop so the hard coded path works for me
with open('3_laps_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # read each line in the file.
    for line in reader:
        samples.append(line)

# # I only train on my laptop so the hard coded path works for me
with open('recover_data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # read each line in the file.
    for line in reader:
        samples.append(line)


def calc_effective_batch_Size(batch_size:int) -> int:
    """
    Calculated the amount of lines that must be effectively loaded from the csv file.
    :param batch_size: the batch size to use for training.
    :return: the effective batch size
    """
    # augmenting the data will double the data set
    if ADD_AUGMENTED:
        assert batch_size % 2 == 0, "Batch size has to be an even number!"
        # so only half as many data must be read
        return int(batch_size / 2)

    # add left and right images to the data set
    if ADD_LR:
        # if the augmented values are added, we effectievly doubled the data set again
        if ADD_AUGMENTED:
            # so only half as many data must be read per batch
            return int(batch_size / 2)
        else:
            # if the augmented data is not included we tripple our data so only a third must be read per batch
            assert batch_size % 3 == 0, "Batch size has to be a number dividable by 3!"
            return int(batch_size / 3)


def generator(samples, batch_size=32):
    """
    Generates batches of training and target data to train on. This generator is designed to generate data
    when it's needed so not the complete data set has to be kept in memory.
    :param samples: the samples read from the csv file (so only the path to the images) and the measurements
    :param batch_size: batch_size to use for training
    :return: a batch to train on
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        effective_batch_size = calc_effective_batch_Size(batch_size)

        # only take as many steps as the effective batch size is
        for offset in range(0, num_samples, effective_batch_size):
            batch_samples = samples[offset:offset+effective_batch_size]

            images = []
            angles = []
            # create the batch
            for batch_sample in batch_samples:
                # use centered data
                name = os.path.join(batch_sample[0])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                # also use flipped data
                if ADD_AUGMENTED:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)

                # also use left and right image data
                if ADD_LR:
                    steering_center = float(line[3])

                    left_img = cv2.imread(line[1])
                    images.append(left_img)
                    steering_left = steering_center + correction
                    angles.append(steering_left)

                    right_img = cv2.imread(line[2])
                    images.append(right_img)
                    steering_right = steering_center - correction
                    angles.append(steering_right)

                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

# split train and validation set 80 /20
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# use generator to create training data
train_generator = generator(train_samples, batch_size=36)
validation_generator = generator(validation_samples, batch_size=36)

model = Sequential()
# trim image to only see section with road
model.add(Cropping2D(cropping=((70, 24), (0, 0)), input_shape=(160, 320, 3)))
# center the pixels
model.add(Lambda(lambda x: x /255.0 - 0.5))
# use the LeNet model to train
model.add(Convolution2D(6, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(16, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
#model.add(Convolution2D(36, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

# model.add(Convolution2D(64, 3, 3))
# model.add(Convolution2D(64, 3, 3))
# model.add(MaxPooling2D((2, 2)))
# model.add(Activation('relu'))

model.add(Flatten())
# model.add(Dense(1164))
model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
model.add(Dense(1))

# use adam optimizer and mse als loss because this is a regression problem
model.compile(loss='mse', optimizer='adam')

# train and create summary
train_size = len(train_samples)
val_size = len(validation_samples)

#adjust the size of the data set to the correct size
if ADD_AUGMENTED:
    # for each training sample we create another augmented one
    train_size += len(train_samples)
    val_size += len(validation_samples)
if ADD_LR:
    # for each training sample we create  2 additional ones
    train_size += 2*len(train_samples)
    val_size += 2*len(validation_samples)

# finally train the data
history_object = model.fit_generator(train_generator, samples_per_epoch=train_size,
                                     validation_data=validation_generator, nb_val_samples=val_size,
                                     nb_epoch=5, verbose=1)
# when done save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
