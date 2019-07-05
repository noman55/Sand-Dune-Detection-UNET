# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 02:22:06 2019

@author: DELL
"""

import numpy as np
import keras
import os
import cv2
from keras import backend as K
from keras.models import load_model



#####################################################

X_train=[]

for i in os.listdir('train lbp/inputs'):
    if '.png' in i:
        X_train=np.append(X_train,i)

Y_train=X_train     
        
X_valid=[]

for i in os.listdir('valid lbp/inputs'):
    if '.png' in i:
        X_valid=np.append(X_valid,i)

Y_valid=X_valid

########################################################
        
images={ 'train':X_train, 
             'validation':X_valid}

masks={ 'train':Y_train, 
        'validation':Y_valid}

#########################################################

print(images['validation'],masks['validation'])
print(images['train'],masks['train'])

#########################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels,path_X,path_Y,shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path_X=path_X
        self.path_Y=path_Y
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        
        X=X/255.0
        Y=Y/255.0
        print("Batch was fed to GPU!")
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))
    
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,:,:,0] = cv2.imread(self.path_X + ID,0)

            # Store class
            Y[i,:,:,0] = cv2.imread(self.path_Y + ID,0)

        return X, Y
    
#########################################################
    
path_to_X_train='train lbp/inputs/'
path_to_Y_train='train lbp/masks/'

path_to_X_valid='valid lbp/inputs/'
path_to_Y_valid='valid lbp/masks/'

batch_size_train=15
batch_size_valid=10


    
training_generator = DataGenerator(list_IDs=images['train'], labels=masks['train'], \
                                   batch_size=batch_size_train, dim=(976,976),\
                                   n_channels=1, path_X=path_to_X_train,path_Y=path_to_Y_train, \
                                   shuffle=True)

validation_generator = DataGenerator(list_IDs=images['validation'], labels=masks['validation'],\
                                     batch_size=batch_size_valid, dim=(976,976),\
                                   n_channels=1, path_X=path_to_X_valid, path_Y=path_to_Y_valid, \
                                   shuffle=True)

#########################################################


#note the indexes can be passed till __len__ only that is (0 to (__len__-1))
#!since floor is used some image are always left ie remainder when divided by batch size! :/

#########################################################

X_test,Y_test=validation_generator.__getitem__(0)

model = load_model('my_model_lbp_10_20_976px.h5')

result=model.predict(X_test)
print(result.shape)

for i in range(10):
    for x in range(976):
        for y in range(976):
            if(result[i,x,y,0]>0.5):
                result[i,x,y,0]=255
            else:
                result[i,x,y,0]=0

img=np.empty((976,976))    
        
for i in range(10):
    img=result[i,:,:,0]
    cv2.imshow("",img)
    cv2.waitKey()
    cv2.destroyAllWindows()









