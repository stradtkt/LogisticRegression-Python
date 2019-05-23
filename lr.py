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
sns.heatmap(train.isnull(), yticklabels=False, char=False, cmap='virdis')