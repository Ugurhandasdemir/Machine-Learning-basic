import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("C:/Users/ugurh/OneDrive/Masaüstü/New Microsoft Excel Çalışma Sayfası.csv", sep=";", encoding='ISO-8859-1')


plt.scatter(df.yil, df.maas)
plt.xlabel("Yil")
plt.ylabel("Maas")
plt.show()

#%% 
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()


#reshape yapmadan önce arrayin shapei (14,) dür ancak sklearn bunu kabul etmez. 
#Sklearn kabul edicegi şekilde olan (14,1) yapmak için reshape kullanırız
x = df.yil.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% prediction
import numpy as np

bo = linear_reg.predict([[0]])
print("bo: ",bo)

bo = linear_reg.intercept_ #y eksenini kestiği nokta (intercept)
print("bo: ",bo)

b1 = linear_reg.coef_ #egim (slope)
print("b1: ",b1)

#y = b0 + b1*x
maas =  23289.1908976 + 1946.58659924*29 #29 yıl deneyim sahibi birinin maaşı
print(maas)

print(linear_reg.predict([[29]])) #sklearn ile 29 yıl deneyimli maas hesaplama

deneyim = np.array([1,2,4,5,7,9,11,13,14,15,16]).reshape(-1,1)

plt.scatter(x, y)
plt.show()

y_head = linear_reg.predict(deneyim)
plt.plot(deneyim, y_head ,color = "red")




