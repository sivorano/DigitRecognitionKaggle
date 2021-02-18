import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
import keras.utils.np_utils


def plotNumberImg(x):
    """
    Reshapes x to a 28x28 img and plots it
    """
    plt.imshow(x.reshape((28,28)))
    plt.show()


# We import the data
# This is the data we train and validate our model on
train = pd.read_csv("train.csv")
# This is the competition test data
test = pd.read_csv("test.csv")


trainNP = train.values

trainLabels = trainNP[:,0]
trainData = trainNP[:,1:]

#We need to make sure we understand our data
print(np.unique(trainLabels,return_counts = True))
print(np.unique(trainData))

# Hence, each number is represented about equaly, and the brighness goes from 0 to 255

# plt.imshow(trainData[0].reshape((28,28)))
# plt.show()                      # 

#Create the actual data sets.


trainRand, valRand = train_test_split(trainNP,test_size = 0.2, random_state = 42)

Xtrain = trainRand[:,1:]
Ytrain = trainRand[:,0]

Xval = valRand[:,1:]
Yval = valRand[:,0]

plotNumberImg(Xtrain[0])

# We normailize the data, here into [0,1]
Xval = Xval/255
Xtrain = Xtrain/255

#We create categorial labels
Ytrain = keras.utils.np_utils.to_categorical(Ytrain, num_classes = 10)
Yval = keras.utils.np_utils.to_categorical(Ytrain, num_classes = 10)





