import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


df = pd.read_csv('countries_world.csv', decimal=',')


df['GDP ($ per capita)'].fillna((df['GDP ($ per capita)'].mean()), inplace=True)
df['Literacy (%)'].fillna((df['Literacy (%)'].mean()), inplace=True)




X=df.iloc[:,8:10]
Y=df.iloc[:,1]
Y=Y.to_frame()

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)





from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,Y_train)


#predicting test results
Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
def accuracy(cm):
    return (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
def error_rate(cm):
    return (cm[0][1]+cm[1][0])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
def sensitivity(cm):
    return cm[0][0]/(cm[0][0]+cm[0][1])
def specificity(cm):
    return cm[1][1]/(cm[1][0]+cm[1][1])
def precision(cm):
    return cm[0][0]/(cm[0][0]+cm[1][0])
def recall(cm):
    return cm[0][0]/(cm[0][0]+cm[0][1])
def f(cm):
    return (2*precision(cm)*recall(cm))/(precision(cm)+recall(cm))

a=accuracy(cm)




from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.2, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('gdp per capita')
plt.ylabel('Literacy(%)')
plt.legend()
plt.show()




