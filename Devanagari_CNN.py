import numpy as np
import cv2
import glob
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array
import xlsxwriter 

def Test_InputDataSet(folder,column):         # Testing input test dataset Images
    row = 0
    for filename in os.listdir(folder):        #Input file for each each folder in Test Dataset
         # Pre-process the image for classification using opencv library
        image = cv2.imread(os.path.join(folder,filename)) 
        image = cv2.resize(image, (32,32))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        l = CNN_Model.predict(image)[0]          # classify the input image 
        worksheet.write(row, column, Character_labels[np.argmax(l)])    # write the output character into excel file       
        row += 1    # increment the value of row 

#Initializing the CNN Model	
CNN_Model = Sequential()

#Add Convolution Layer
CNN_Model.add(Convolution2D(filters = 32,kernel_size = (3,3),strides = 1,activation = "relu",input_shape = (32,32,1)))
CNN_Model.add(Convolution2D(filters = 32,kernel_size = (3,3),strides = 1,activation = "relu"))

#Add Pooling Layer
CNN_Model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="same"))

# Add Convolution Layer		
CNN_Model.add(Convolution2D(filters = 64,kernel_size = (3,3),strides = 1,activation = "relu"))
CNN_Model.add(Convolution2D(filters = 64,kernel_size = (3,3),strides= 1,activation = "relu"))

#Add Pooling Layer
CNN_Model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="same"))			

#Add a dropout layer to overcome the problem of overfitting to some extent.			
CNN_Model.add(Dropout(0.3))

#Add Flattening Layer
CNN_Model.add(Flatten())

#Add Hidden layers
CNN_Model.add(Dense(128,activation = "relu",kernel_initializer = "uniform"))			
CNN_Model.add(Dense(64,activation = "relu",kernel_initializer = "uniform"))			

#Add Output Layer
output_layer = Dense(46, activation= "softmax", kernel_initializer='uniform')
CNN_Model.add(output_layer)

#Compiling CNN_Model
CNN_Model.compile(optimizer = "adam",loss = "categorical_crossentropy", metrics = ["accuracy"])

#Generating Image Data		
train_Data = ImageDataGenerator(rescale = 1.0/255,shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True)
#Fitting images to CNN Model
train_Set = train_Data.flow_from_directory("/home/Desktop/DevanagariHandwrittenCharacterDataset/Train",target_size = (32,32),batch_size = 32,color_mode = "grayscale",class_mode = 'categorical')

# Training CNN_Model
CNN_Model.fit_generator(train_Set,steps_per_epoch=train_Set.n//train_Set.batch_size, epochs=10)

#Define the Devanagari Characters using unicode
Character_labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']

#Creating excel sheet and writing outputs into it
workbook = xlsxwriter.Workbook('/home/Desktop/Output_TestData.xlsx') 
worksheet = workbook.add_worksheet()

folders = glob.glob('/home/Desktop/DevanagariHandwrittenCharacterDataset/Test/*') #Test Data Set folder
column=0
for folder in folders:          #Taking input images from subfolders of Test image Data Set
    Test_InputDataSet(folder,column)
    column=column+1

workbook.close()








