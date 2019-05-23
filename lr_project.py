# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:42:53 2019

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()

ad_data['Age'].plot.hist(bins=35)

sns.jointplot(x='Age', y='Area Income', data=ad_data)

sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde')

sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)

sns.pairplot(ad_data, hue='Clicked on Ad')

from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)