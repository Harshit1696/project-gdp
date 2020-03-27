

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


df = pd.read_csv('countries of the world.csv', decimal=',')
data=df.copy()
data=data.drop(['GDP ($ per capita)'], axis=1)
data['Phones (per 1000)'].fillna((df['Phones (per 1000)'].mean()), inplace=True)

X=data.iloc[:,[9]]
X=X.to_frame()

df['GDP ($ per capita)'].fillna((df['GDP ($ per capita)'].mean()), inplace=True)
Y=df.iloc[:, 8]
Y=Y.to_frame()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=114)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

print('coefficent\n' , regressor.coef_)

print('intercept' , regressor.intercept_)

Y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error , r2_score
print("mean squared error: {}".format(mean_squared_error(Y_test,Y_pred)))
print("r2 score: {}".format(r2_score(Y_test,Y_pred)))
print("Mean absolute error:",np.sqrt(metrics.mean_absolute_error(Y_test, Y_pred)))

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("gdp vs phones per 1000")
plt.xlabel("phones per 1000")
plt.ylabel("gdp")















