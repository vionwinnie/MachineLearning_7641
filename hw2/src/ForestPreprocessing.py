# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:15:49 2019

@author: USER
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

##===========================##
##   Checking Missing Value  ##
##===========================##

df = pd.read_csv('../data/winequality-white.csv',sep=';')

#categorize wine quality
bins = (2,6.5,9)
group_names = [0,1]
categories = pd.cut(df['quality'], bins, labels = group_names)
df['quality'] = categories

## Divide into features and output
X = df.loc[:,'fixed acidity':'alcohol']
y = np.array(df.loc[:,'quality'])

## splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


## Standardize the data before training the model
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(df.columns[:-1])


train_df = np.hstack([X_train,y_train.reshape(y_train.shape[0],1)])
test_df = np.hstack([X_test,y_test.reshape(y_test.shape[0],1)])

np.savetxt("../data/wine_train.csv", train_df, delimiter=",")
np.savetxt("../data/wine_test.csv", test_df, delimiter=",")

print('finished preparing the data')
