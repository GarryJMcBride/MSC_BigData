import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

premierleagueplayers = pd.read_csv("Statistical Analysis of Fantasy Premier League\Data\FPL_2018_19_Wk7.csv")


premierleagueplayers = premierleagueplayers.drop(columns=['Name', 'Team', 'Position'])
print(premierleagueplayers)


X = premierleagueplayers.values[:, 8:16]
Y = premierleagueplayers.values[:, 3]


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.30)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import metrics
premierleagueplayers.keys()
premierleagueplayers.data.shape
print(premierleagueplayers.feature_names)
print(premierleagueplayers.DESCR)
bos = pd.DataFrame(premierleagueplayers.data)
bos.head()
bos.columns = premierleagueplayers.feature_names
bos.head()


print("LOGISTIC REGRESSION")
print("**************************************")
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("Decision Tree")
print("**************************************")
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("KNN")
print("**************************************")
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("Naive Bayes")
print("**************************************")
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(premierleagueplayers.data, premierleagueplayers.target, test_size = 0.25)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import metrics


premierleagueplayers.keys()
premierleagueplayers.data.shape
print(premierleagueplayers.feature_names)
print(premierleagueplayers.DESCR)
bos = pd.DataFrame(premierleagueplayers.data)
bos.head()
bos.columns = premierleagueplayers.feature_names
bos.head()
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(bos, premierleagueplayers.target, test_size=0.25)
lm = LinearRegression()
lm.fit(X_train, Y_train)
lm.fit(bos, premierleagueplayers.target)
print(lm.intercept_)
print(lm.coef_)
pd.DataFrame(list(zip(bos.columns, lm.coef_)), columns=['Features', 'Coefficients'])
plt.scatter(bos.RM, premierleagueplayers.target)
plt.xlabel('Average number of rooms')
plt.ylabel('House Price')
plt.title('relationship between number of rooms and price')
plt.show()
predict = lm.predict(X_test)
print(metrics.mean_squared_error(Y_test, predict))

from sklearn.cross_validation import KFold

kfold = KFold(10)
lm = LinearRegression()
results = sklearn.cross_validation.cross_val_score(lm, bos, premierleagueplayers.target, cv=kfold, scoring="mean_absolute_error")
print(results.mean())
results = sklearn.cross_validation.cross_val_score(lm, bos, premierleagueplayers.target, cv=kfold, scoring="mean_squared_error")
print(results.mean())
results = sklearn.cross_validation.cross_val_score(lm, bos, premierleagueplayers.target, cv=kfold, scoring="r2")
print(results.mean())

