import csv
import cv2
import numpy as np

lines = []

# Open the csv log file and read the file name of the figure
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
        
for line in lines:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        token = source_path.split('/')
        filename = token[-1]
        local_path = './data/IMG/' + filename  # in order to run the code on AWS
        image = cv2.imread(local_path)
        images.append(image)
    correction = 0.2 
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    
print(len(measurements))
    
augmented_images = []
augmented_measurements = []

# flip the images to generate more images
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
    
print(len(augmented_measurements))
    
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape)
print(X_train[1].shape)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Nvidia End to End Self-driving Car CNN
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print('model ready')

model.compile(optimizer='adam', loss = 'mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=128, nb_epoch = 5)

model.save('model.h5')
print('Model Saved!')
