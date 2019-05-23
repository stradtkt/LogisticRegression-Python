# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:49:34 2019

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)

sns.countplot(x='Survived', hue='Sex', data=train)
sns.countplot(x='Survived', hue='Pclass', data=train)

sns.distplot(train['Age'].dropna(), kde=False, bins=30)

train['Age'].plot.hist(bins=35)

train.info()

sns.countplot(x='SibSp', data=train)

train['Fare'].hist(bins=30, figsize=(10,4))

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age