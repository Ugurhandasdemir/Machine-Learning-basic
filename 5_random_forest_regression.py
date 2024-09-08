# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:29:36 2024

@author: ugurh
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv("5_random_forest_regression.csv", sep = ";", header = None )
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

rf = RandomForestRegressor()
rf.fit(x,y)

print("6.8 seviyesinde fiyat : ", rf.predict([[6.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x, y, color ="red")
plt.plot(x_, y_head, color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show