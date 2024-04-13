#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


#In the DyslexiaML notebook, we found that RandomForest with GridSearch is best fit for the given dataset.
#This model gives the most accurate predictions.
#In this notebook, we will create only a RandomForest model, which will then be used to make final predictions.


# In[22]:


#Reading the dataset
data=pd.read_csv('labeled_dysx.csv')
#Value to be predicted by the model.
y=data.Label 
#Input taken by the model.
X=data.drop(['Label'],axis=1) 
data.head()


# In[23]:


#In the given data, the label is the indication for whether the person has dislexia or not.
#Label = 0 means that there is a high chance that the person has dislexia.
#Label = 1 means that there is a moderte chance that the person has dislexia.
#Label = 2 means that there is a low chance that the person has dislexia.
#The Survey_Score is calculated on the basis of the answers to the quiz given by the applicant.


# In[24]:


#Creating the test and train data sets for the given data.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=10)


# In[25]:


#StandardScalar is used for preprocessing of data.
#'copy' is False, which means copies are avoid and inplace scaling is done instead.
sc=StandardScaler(copy=False)
sc.fit_transform(X_train)
sc.transform(X_test)


# Creating RandomForest model with GridSearch

# In[26]:


#Creating a list of possible n_estimators.
n_est = {'n_estimators' : [10,100,500,1000]}
#Creating a RandomForest model using the value of n_estimators given by GridSearch for best result.
model = GridSearchCV(RandomForestClassifier(random_state=0),n_est,scoring='f1_macro')
#Training the model
model.fit(X_train, y_train)
#Making predictions using the model.
predictions = model.predict(X_test)
#Printing the value of n_estimator used in the model.
#This value provides the most accurate predictions for our dataset.
print('Best value of n_estimator for RandomForest model is:')
print(model.best_params_)

