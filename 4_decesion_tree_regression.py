import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("C:/Users/ugurh/OneDrive/Masaüstü/Yeni klasör/Belgeler/spyder/machine_learning/4_decesion_tree_regression.csv", sep = ";", header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# decesion tree regressipn

reg_tree = DecisionTreeRegressor()
reg_tree.fit(x,y)

print(reg_tree.predict([[4.4]]))

x_ = np.arange(min(x),max(x), 0.01).reshape(-1,1) # minumun x değerinden maximum x değerine 0.01 artıcka şekilde bir array oluşturuyor.
y_head = reg_tree.predict(x_)
print("y head : ", y_head)


# visualize

plt.scatter(x, y, color = "red")
plt.plot(x_, y_head, color="blue")
plt.xlabel("Tribun level")
plt.ylabel("price")
plt.show()