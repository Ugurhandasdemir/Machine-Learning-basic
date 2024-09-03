import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/ugurh/OneDrive/Masaüstü/Yeni klasör/Belgeler/spyder/machine_learning/3_polynomial_regression.csv", sep = ";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x, y)
plt.xlabel("Araba fiyat")
plt.ylabel("Araba hiz")
plt.show()

#%% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)
y_head = lr.predict(x)

plt.plot(x, y_head, color = "red", label="linear")
plt.show()

print("10 milyonluk araba fiyati ",lr.predict([[10000]]))
#grafikte gördüğümüz gibi liner olmayan veri değelerlerinde liner regression kullanırsak prediclerimiz hataları sonuçlar verir.
#bu durumlarda polynomial regression kullanılmalıdır.

#%% polynomial regression  y = bo + b1*x + b2*x^2 + b3*x^3 ... bn*x^n 
from sklearn.preprocessing import PolynomialFeatures

pl = PolynomialFeatures(degree=4) #grafiğin derecesinin 2 olduğunu belirtiyoruz
x_pl = pl.fit_transform(x) # fit_transform kullanarak x^2' yi elde ediyoruz.

lr2 = LinearRegression()
lr2.fit(x_pl,y)

y_head2 = lr2.predict(x_pl)

plt.plot(x,y_head2, color = "green", label = "polynomial")
plt.show()
