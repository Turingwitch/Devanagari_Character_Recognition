# Devanagari_Character_Recognition
The dataset used is Devanagari Character Dataset available at https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset

## Data Set Information:
> Image Format: .png\
> Resolution: 32 by 32 \
> Data Type: GrayScale Image\
> Total_Classes: 46\
> Dataset Size : 92000\
> Training Dataset Size: 78200(46 classes(1700 images in each class))\
> Testing Dataset Size: 13800(46 classes(300 images in each class))\
> Actual character is centered within 28 by 28 pixel, padding of 2 pixel is added on all four sides of actual character.

## Convolutional Neural Network Model:

>Convolutional Neural Network is implemented in keras for multiclass classification of Devanagari Handwritten Character Dataset with training accuracy 97.02% (10 epochs).\
>For processing input Test dataset, opencv python library is used.\
>Output_TestData.xlsx excel file contains outputs for the test data set with 46 columns(46 character class folders) and 300 rows(300 input images in each folder). Each column value represents the predicted output character corresponding to each folder in Test Data Set.\
>Unicode for "क्ष , त्र , ज्ञ" is not available. Instead of "क्ष , त्र , ज्ञ" ,'ksha','tra','gya' is used for representing these characters.



