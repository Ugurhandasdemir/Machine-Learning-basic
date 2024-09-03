import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


df = pd.read_csv("C:/Users/ugurh/OneDrive/Masaüstü/Yeni klasör/Belgeler/spyder/machine_learning/multiple_linear_regression.csv", sep = ";")

x = df.iloc[:,[0,2]].values
y= df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

bo = multiple_linear_regression.intercept_
b1 , b2 = multiple_linear_regression.coef_[0]

print("bo: ", bo, "b1: ", b1, "b2: ",b2)

test_value = np.array([[10,45],[6,45]])
multiple_linear_regression.predict(test_value)


