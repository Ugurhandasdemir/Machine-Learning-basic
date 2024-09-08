# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:48:49 2024

@author: ugurh
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/ugurh/OneDrive/Masaüstü/Yeni klasör/Belgeler/spyder/machine_learning/7_CarPrice_Assignment.csv")
#%%Linear Regression 
horsepower = df.iloc[:,21].values.reshape(-1,1) 
car_price = df.iloc[:,25].values.reshape(-1,1)

lr = LinearRegression()
lr.fit(horsepower, car_price)

print("145 HP için tahmin edilen fiyat: ", lr.predict([[145]]))
y_head = lr.predict(horsepower)

print("Linear regression r2 score: ", r2_score(car_price, y_head))


plt.figure()
plt.scatter(horsepower,car_price , color = "red",label="Gerçek Değerler")
plt.plot(horsepower,y_head, color = "green")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Linear Regression")

plt.show()

#%%multiple linear regression
mlr = LinearRegression()

car_lenghth_and_car_witdth = df.iloc[:,[10,11]].values

mlr.fit(car_lenghth_and_car_witdth,car_price)
y_head_mlr = mlr.predict(car_lenghth_and_car_witdth)

#print("Model tahminleri: ", y_head )
print("Multiple Linear Regression r2 score: " ,r2_score(car_price, y_head_mlr) )

print(" 188,8 cm uzunlugunda ve 87.8 genisligindeki araba fiyatı tahmini", mlr.predict([[188.8,87.6]]))


#%%polynominal regressino

pl = PolynomialFeatures(degree=3)
horsepower_pl = pl.fit_transform(horsepower)

lr2 = LinearRegression()
lr2.fit(horsepower_pl,car_price)

y_head2 = lr2.predict(horsepower_pl)

print("Polynomial Regression r2: ",r2_score(car_price, y_head2))

plt.figure()
plt.scatter(horsepower, car_price, color="red", label="Gerçek Değerler")
plt.plot(horsepower, y_head2, color="blue", label="Polinomial Regresyon (Degree=3)")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Polynominal Regression")

plt.show()


#%%decesion tree

dec_tree = DecisionTreeRegressor()
engine_size=df.iloc[:,16].values.reshape(-1,1)

dec_tree.fit(engine_size, car_price)

y_head_decesion_tree = dec_tree.predict(engine_size)

print("Decesion tree regression r2", r2_score(car_price,y_head_decesion_tree ))

engine_size_ = np.arange(min(engine_size),max(engine_size), 0.01 ).reshape(-1,1)
y_head_decesion_tree_ = dec_tree.predict(engine_size_)

plt.figure()
plt.scatter(engine_size, car_price, color="red", label="Gerçek Değerler")
plt.plot(engine_size_, y_head_decesion_tree_, color="blue", label="Decision Tree Regression")
plt.xlabel("engine size")
plt.ylabel("price")
plt.title("Decision Tree Regression")

plt.show()

#%%Random Forest

car_height = df.iloc[:,12].values.reshape(-1,1)
rf = RandomForestRegressor()

rf.fit(car_height,car_price)
y_head_car_height = rf.predict(car_height)
print("Random Forest r2 :" , r2_score(car_price,y_head_car_height))

car_height_ =np.arange(min(car_height), max(car_height),0.01).reshape(-1,1)
y_head_car_height_ = rf.predict(car_height_)


plt.figure()
plt.scatter(car_height,car_price,color = "purple", label="Gerçek Değerler")
plt.plot(car_height_, y_head_car_height_, color= "orange", label="Random Forest Regression")
plt.xlabel("Car height")
plt.ylabel("Car price")
plt.title("Random Forest Regression")
plt.show()



