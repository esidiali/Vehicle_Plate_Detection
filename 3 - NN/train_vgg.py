import pandas as pd
import numpy as np
import os

import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import cv2
#### the same preprocessing functions, as in RandomForests model, are used here.


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
os.environ["CUDA_VISIBLE_DEVICES"]="0"
new_shape = (224, 224)

# Creating the model
print("\nCreating and compiling the model...")
base_model=VGG16(weights='imagenet', include_top=False)
 #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x)
preds=Dense(4,activation='relu')(x) 

model=Model(inputs=base_model.input,outputs=preds)

stop_training = 20

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:stop_training]:
    layer.trainable=False
for layer in model.layers[stop_training:]:
    layer.trainable=True
    
print(model.summary())

model.compile(optimizer='Adam',loss='mean_squared_error',metrics=['mae'])

def process(input_file):
    with open(input_file, 'r') as f:
        a = f.readlines()[0]
    a = a.split("\t")
    name = a[0]
    bbox = list(map(int, a[1:5]))
    img = cv2.imread(input_file[:-4] + '.jpg')
    return img, bbox

def resize_im_box(image, box, new_shape):
    resized_img = cv2.resize(image, new_shape)
    
    scale_y = new_shape[0] / image.shape[0]
    scale_x = new_shape[1] / image.shape[1]
    
    resized_box = box.copy()
    resized_box[0] = int(box[0]*scale_x)
    resized_box[2] = int(box[2]*scale_x)
    resized_box[1] = int(box[1]*scale_y)
    resized_box[3] = int(box[3]*scale_y)
    
    return resized_img, resized_box
   

print("\nLoading EU images...")
eu_img_buffer = []
eu_box_buffer = []

for input_file in os.listdir("endtoend/eu/"):
    if input_file.endswith('.txt'):
        img, box = process("endtoend/eu/" + input_file)
        img, box = resize_im_box(img, box, new_shape = new_shape)
        eu_img_buffer.append(img)
        eu_box_buffer.append(box)
        
us_img_buffer = []
us_box_buffer = []

print("\nLoading US images...")

for input_file in os.listdir("endtoend/us/"):
    if input_file.endswith('.txt'):
        img, box = process("endtoend/us/" + input_file)
        img, box = resize_im_box(img, box, new_shape = new_shape)
        us_img_buffer.append(img)
        us_box_buffer.append(box)

print("\nLoading BR images...")

br_img_buffer = []
br_box_buffer = []

for input_file in os.listdir("endtoend/br/"):
    if input_file.endswith('.txt'):
        img, box = process("endtoend/br/" + input_file)
        img, box = resize_im_box(img, box, new_shape = new_shape)
        br_img_buffer.append(img)
        br_box_buffer.append(box)


print("\n Loading data done")
remove_for_vis = 5
img_data_to_consider = eu_img_buffer[:-remove_for_vis] + us_img_buffer[:-remove_for_vis] + br_img_buffer[:-remove_for_vis]
box_data_to_consider = eu_box_buffer[:-remove_for_vis] + us_box_buffer[:-remove_for_vis] + br_box_buffer[:-remove_for_vis]

def process_line(line):
    splits = line.split('\t')
    name = splits[0]
    box = list(map(int, splits[1:-1]))
    img = cv2.imread("augmented/" + name)
    return img, box
    
print("\nLoading augmented images...\n") 

with open("augmented/augmented.txt", "r") as my_file:
    lines = my_file.readlines()
    

for i in range(len(lines)):
    line = lines[i]
    img, box = process_line(line)
    img, box = resize_im_box(img, box, new_shape = new_shape)
    img_data_to_consider.append(img)
    box_data_to_consider.append(box)

#img_viz = eu_img_buffer[-remove_for_vis:] + us_img_buffer[-remove_for_vis:] + br_img_buffer[-remove_for_vis:]
#box_viz = eu_box_buffer[-remove_for_vis:] + us_box_buffer[-remove_for_vis:] + br_box_buffer[-remove_for_vis:]


print("Train-test split")

X_train, X_test, y_train, y_test = train_test_split(np.array(img_data_to_consider), np.array(box_data_to_consider), test_size = 0.1, random_state = 42)

print("\nX_train shape : ", X_train.shape)
print("y_train shape : ", y_train.shape)
print("X_test shape : ", X_test.shape)
print("y_test shape : ", y_test.shape)

print('\n')

print("Training...\n")

nb_epochs = 1000
bsize = 64
filepath = "saved_models/my_VGG_model_{}_bsize_{}".format(nb_epochs, bsize)
cback = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit(X_train, y_train, epochs=nb_epochs, batch_size=bsize, validation_data = (X_test, y_test), callbacks=[cback])

print("Saving the model...")

model.save("my_VGG_model_{}_bsize_{}".format(nb_epochs, bsize))

