# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:50:48 2024

@author: ugurh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:29:36 2024

@author: ugurh
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv("5_random_forest_regression.csv", sep = ";", header = None )
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

rf = RandomForestRegressor()
rf.fit(x,y)


#random forest r-squared
y_head = rf.predict(x)
print("r_score random forest: ", r2_score(y ,y_head))

#%% linear forest r-squared
linear_reg = LinearRegression()


df1 = pd.read_csv("C:/Users/ugurh/OneDrive/Masaüstü/Yeni klasör/Belgeler/spyder/machine_learning/1_linear_regression.csv", sep=";", encoding='ISO-8859-1')

x1 = df1.yil.values.reshape(-1,1)
y1 = df1.maas.values.reshape(-1,1)

linear_reg.fit(x1,y1)

y_head1 = linear_reg.predict(x1)
print("r_score linear", r2_score(y1, y_head1))

