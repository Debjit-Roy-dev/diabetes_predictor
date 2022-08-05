# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 19:32:08 2022

@author: DEBJIT
"""

import numpy as np
import pandas as pd
"""standadisation of data to a common range"""

from sklearn.preprocessing import StandardScaler 
""" splitting of data"""
from sklearn.model_selection import train_test_split

"""trainning model"""
from sklearn import svm

"""accuracy"""

from sklearn.metrics import accuracy_score


 
"""Data collection and analysis"""


#loading dataset
diabetes_dataset=pd.read_csv('D:\ML projects\diabetes.csv')

#separating data and lables(input variables and output)

X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

"""data standardization"""
#values of diff variables are in diff ranges needs to be processed

scaler=StandardScaler()
scaler.fit(X)
#get all variable data in same range(0,1)
standardized_data=scaler.transform(X)
X=standardized_data


"""splitting"""
#stratify means distribute in same proportion
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.25,stratify=Y,random_state=2)


"""trainning"""
#model
classifier=svm.SVC(kernel='linear')

#trainning svm classifier

classifier.fit(X_train,Y_train)


"""Model Evaluation"""

#accuracy score on train data

X_train_predict=classifier.predict(X_train)
trainning_data_accuracy=accuracy_score(X_train_predict,Y_train)

print("accuracy of trainning data",trainning_data_accuracy)


#accuracy score on test data

X_test_predict=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_predict,Y_test)

print("accuracy of testing data",test_data_accuracy)


"""making a predictive system"""

input_data=tuple(input("enter values of Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age:").split(","))#(4,110,92,0,0,37.6,0.191,30)
#change to numpy array
input_data_as_nparr=np.asarray(input_data)

#reshape data as we are testing for 1 instance
input_data_reshaped=input_data_as_nparr.reshape(1,-1)

#need to standardise as model trained on standardized data

std_data=scaler.transform(input_data_reshaped)

"""prediction"""

prediction=classifier.predict(std_data)
if(prediction[0]==0):
    print("not diabetic")
else:
    print("diabetic")































