#!/usr/bin/env python
# coding: utf-8

# ### This tutorial is to teach you how to use Scikit-learn for Machine Learning. We used CLASSIFICATION - SVM Classifier

# In[32]:


import matplotlib.pyplot as plt

from sklearn import datasets    #example, iris, numbers exisitng datasets in scikit-learn. 
from sklearn import svm         #svm, support vector machine is used for classification for supervised learning 


# In[33]:


#import and load digit datasets
digits = datasets.load_digits()
print(digits.images[2])  #just an example
len(digits.target)


# In[48]:


# now lets develop a Classifier 
clf = svm.SVC(gamma=0.001, C=100)   #The higher the gamma value it tries to exactly fit the training data set gammas

#split the data into x,y
x,y =digits.data[:-10], digits.target[:-10]    #this data and targets gonna be our TEST set


# In[49]:


#model fitting. Model in this case is the classifier
clf.fit(x,y)


# In[53]:


print('Prediction of last:', clf.predict(digits.data[[-4]]))   #it tries to predict the last digit
plt.imshow(digits.images[-4])    #display the real image, not the predicted one.


# In[ ]:


import matplotlib
import numpy
import sklearn

