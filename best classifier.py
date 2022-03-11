# Logistic Regression

## Importing the libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""## Importing the dataset"""

dataset = pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/Online%20Shoppers%20Intention.csv')

dataset.apply(pd.Series.nunique)

dataset.head(10)

dataset.info()

f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(dataset.corr(), cmap="GnBu", annot=True, linewidths=0.5, fmt= '.1f',ax=ax)
plt.show()

sns.countplot(dataset.Revenue, palette="PRGn")
plt.title("Revenue",fontsize=15)
plt.show()

dataset.shape

print(dataset.isna().sum())
dataset = dataset.dropna()

dataset = dataset.drop('Month', 1)

dataset.head()

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

X[:,-1] = le.fit_transform(X[:,-1])
X

le2 = LabelEncoder()
X[:,-2] = le2.fit_transform(X[:,-2])
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=101)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTclassifier = DecisionTreeClassifier(random_state = 0, criterion='entropy')
DTclassifier.fit(X_train, y_train)

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
GBclassifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
GBclassifier.fit(X_train, y_train)

y_predlog = logmodel.predict(X_test)
y_predDT = DTclassifier.predict(X_test)
y_predGB = GBclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cmlog = confusion_matrix(y_test, y_predlog)
acclog = accuracy_score(y_test, y_predlog)
print("ACCURACY OF LOGISTIC REGRESSION:",acclog)

print("CONFUSION MATRIX OF LOGISTIC REGRESSION")
print(cmlog)

ax = sns.heatmap(cmlog/np.sum(cmlog), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix for Logistic Regression\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.metrics import classification_report
print("Classification Report of Logistic Regression")
print(classification_report(y_test, y_predlog))

cmdt = confusion_matrix(y_test, y_predDT)
accdt = accuracy_score(y_test, y_predDT)
print("ACCURACY OF DECISION TREE:",accdt)

print("CONFUSION MATRIX OF DECISION TREE")
print(cmdt)

ax = sns.heatmap(cmdt/np.sum(cmdt), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix for Decission Tree\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.metrics import classification_report
print("Classification Report of Decision Tree")
print(classification_report(y_test, y_predDT))

cmgb = confusion_matrix(y_test, y_predGB)
accgb = accuracy_score(y_test, y_predGB)
print("ACCURACY OF GRADIENT BOOSTING",accgb)

print("CONFUSION MATRIX OF GRADIENT BOOSTING")
print(cmgb)

ax = sns.heatmap(cmgb/np.sum(cmgb), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix for Gradient Boosting\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.metrics import classification_report
print("Classification Report of Gradient Boosting")
print(classification_report(y_test, y_predGB))
