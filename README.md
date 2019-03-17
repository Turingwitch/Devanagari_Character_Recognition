# Devanagari_Character_Recognition
The dataset used is part of the Devanagari Character Dataset available at [https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset]

## Data Set Information:
> Image Format: .png\
> Resolution: 32 by 32 \
> Data Type: GrayScale Image\
> Total_Classes: 46\
> Dataset Size : 92000\
> Training Dataset Size: 78200(46 classes(1700 images in each class))\
> Testing Dataset Size: 13800(46 classes(300 images in each class))\
> Actual character is centered within 28 by 28 pixel, padding of 2 pixel is added on all four sides of actual character.

## Training/Evaluation:

>Convolutional Neural Network is implemented in keras for multiclass classification of Devanagari Handwritten Character Dataset with training accuracy 96.938%.\
>For processing input Test dataset, opencv python library is used.\
>Output of Test dataset is stored in An excel files containing outputs for the test data set. 



