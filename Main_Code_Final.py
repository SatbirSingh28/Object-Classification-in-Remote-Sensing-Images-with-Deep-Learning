#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### ENEL 645 Final Project
##### Dec 20, 2020
##### Authors: Mirza Danish Baig, Maheen Hafeez, and Satbir Singh


#Importing the relevant libraries

from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import keras
from keras.datasets import mnist
import os
from keras.models import Sequential

print('Importing libraries successful!')


# In[3]:


#Creation of the following variables/arrays will help us reduce the image size from 
# the Dior dataset and the images will be resized from 800 x 800 to 32x32

img_rows, img_cols = 32,32
train_images = []
test_images = []
mydata_images=[]

print('Creating variables successful!')


# In[4]:


# This is where all the cropped images will be and resized to 32 x 32 and saved in this variable
input_shape = (32,32,3)


# In[5]:


#The following code will read all the cropped training images from the Dior dataset, resize them, and 
#store them in the variable train_images

for i in range(1, 29249):
    img1 = Image.open(os.path.join('cropped_train_images', str(i) + ".jpg"))
    newsize = (32, 32)
    im1 = img1.resize(newsize)
    image1 = np.asarray(im1) / 255.0
    train_images.append(image1)
    
print('Cropped Training Images imported and resized into array successfully!')


# In[6]:


#The following code will read all the cropped testing images from the Dior dataset, resize them, and 
#store them in the variable test_images

for i in range(1, 38382):
    img3 = Image.open(os.path.join('cropped_test_images', str(i) + ".jpg"))
    newsize = (32, 32)
    im3 = img3.resize(newsize)
    image3 = np.asarray(im3) / 255.0
    test_images.append(image3)
    
print('Cropped Testing Images imported and resized into array successfully!')


# In[7]:


#The following code will read all the cropped testing images from the Sentinel-2 dataset, resize them, and 
#store them in the variable mydata_images

for i in range(1, 43):
    img4 = Image.open(
        os.path.join('cropped_Images_MyData', str(i) + ".jpg"))
    newsize = (32, 32)
    im4 = img4.resize(newsize)
    image4 = np.asarray(im4) / 255.0
    mydata_images.append(image4)
    
print ('Cropped Testing Images from the Sentinel-2 dataset imported and resized into array successfully!')


# In[12]:


#The following code converts the variables created above into numpy arrays

train_image_database = np.array(train_images)
test_image_database = np.array(test_images)
mydata_image_database=np.array(mydata_images)

#The following code creates the target variables for the datasets

train_label = []
test_label = []
mydatacsv_label=[]

print('Numpy arrays and target variables creation successful!')


# In[13]:


# The following code will open the csv file with the training data and encodes the classes as follows:
# ship == 0
# stadium == 1
# bridge == 2

# These classes are then stored in a variable train_label1

with open('Train_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 1

    for row in csv_reader:
        if (line_count == 29249):
            break

        else:
            label = str(row[4])
            if (label == "ship"):
                label = 0
            if (label == "stadium"):
                label = 1
            if (label == "bridge"):
                label = 2
            train_label.append(label)
            line_count += 1

    train_label = np.array(train_label)
train_label1=train_label

print('Encoding for the training data created sucessfully!')


# In[14]:


# The following code will open the csv file with the testing data of the Dior dataset and 
# encodes the classes as follows:
# ship == 0
# stadium == 1
# bridge == 2

# These classes are then stored in a variable test_label

with open('Test_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 1

    for row in csv_reader:
        if (line_count == 38382):
            break

        else:
            label = str(row[4])
            if (label == "ship"):
                label = 0
            if (label == "stadium"):
                label = 1
            if (label == "bridge"):
                label = 2
            test_label.append(label)
            line_count += 1

    test_label = np.array(test_label)
test_label=test_label

print('Encoding for the testing data created sucessfully!')


# In[15]:


# The following code will open the csv file with the testing data of the Sentinel-2 dataset and 
# encodes the classes as follows:
# ship == 0
# stadium == 1
# bridge == 2

# These classes are then stored in a variable mydatacsv_label

with open('Mydata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 1

    for row in csv_reader:
        if (line_count == 43):
            break

        else:
            label = str(row[4])
            if (label == "ship"):
                label = 0
            if (label == "stadium"):
                label = 1
            if (label == "Bridge"):
                label = 2
            mydatacsv_label.append(label)
            line_count += 1

mydatacsv_label = np.array(mydatacsv_label)

print('Encoding for the Sentinel-2 testing data created sucessfully!')


# In[17]:


#The following code creates the training variables for the three datasets above

x_train = train_image_database
y_train = train_label
x_test = test_image_database
y_test = test_label
J_data=mydata_image_database
J_label=mydatacsv_label

print('Training variables created successfully!')


# In[19]:


#The following code shows the sizes of the x_train variable

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[27]:


#The following code creates the CNN Model


# This will convert the class vectors to binary class matrices and resize them to 2 dimensional variables

y_train = keras.utils.to_categorical(y_train, classes_num)
y_test = keras.utils.to_categorical(y_test, classes_num)

# The following creates the CNN Model. The code lines below are further elaborated in the Report.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # features as 1D
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))
model.summary()


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

print('CNN Model Created Successfully!')


# In[32]:


# The following code will fit the CNN model with the Dior training dataset and evaluate the score

# Here we define the parameters of the CNN Model

size_batch = 128
classes_num =3
epochs = 1

history = model.fit(x_train, y_train,batch_size=size_batch,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('The model is fit with data successfully!')


# In[ ]:




