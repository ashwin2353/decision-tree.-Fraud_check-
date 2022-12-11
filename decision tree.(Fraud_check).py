# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:48:38 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("Fraud_check.csv")

df.shape
df.dtypes
df.head(10)

df['Taxable.Income'].value_counts()
df['Taxable.Income'].max()

#===================================================
# box plot

import matplotlib.pyplot as plt
plt.boxplot(df['City.Population'],vert=False)
import numpy as np
Q1 = np.percentile(df['City.Population'],25)
Q2 = np.percentile(df['City.Population'],50)
Q3 = np.percentile(df['City.Population'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['City.Population'] < LW) | (df['City.Population'] > UW)]

len(df[(df['City.Population'] < LW) | (df['City.Population'] > UW)])
# 0 out layers
#===================================================
import matplotlib.pyplot as plt
plt.boxplot(df['Work.Experience'],vert=False)
import numpy as np
Q1 = np.percentile(df['Work.Experience'],25)
Q2 = np.percentile(df['Work.Experience'],50)
Q3 = np.percentile(df['Work.Experience'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Work.Experience'] < LW) | (df['Work.Experience'] > UW)]

len(df[(df['Work.Experience'] < LW) | (df['Work.Experience'] > UW)])
# 0 out layers
#====================================================

plt.boxplot(df['Taxable.Income'],vert=False)

Q1 = np.percentile(df['Taxable.Income'],25)
Q2 = np.percentile(df['Taxable.Income'],50)
Q3 = np.percentile(df['Taxable.Income'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Taxable.Income'] < LW) | (df['Taxable.Income'] > UW)]
len(df[(df['Taxable.Income'] < LW) | (df['Taxable.Income'] > UW)])

# 0 out layers

#================================================
# convertid numarical varialble into categorical variable
pd.cut(df['Taxable.Income'], bins=[0,30000,99619], labels=("Risky","Good")).head(30)
df['Taxable.Income'] = pd.cut(df['Taxable.Income'], bins=[0,30000,99619], labels=("Risky","Good"))

df['Taxable.Income']
df["Taxable.Income"].value_counts()
df.dtypes
#==================================================
# Label encoding

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Undergrad'] = LE.fit_transform(df['Undergrad'])
df['Marital.Status'] = LE.fit_transform(df['Marital.Status'])
df['Urban'] = LE.fit_transform(df['Urban'])

list(df)
df.info()

#==================================================
# deviding the variables into X and Y 

Y = df['Taxable.Income']
X = df[['Undergrad','Marital.Status','City.Population','Work.Experience','Urban']]

#==================================================
# Data partision
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)
 
X_train.shape
X_test.shape
Y_train.shape
Y_test .shape
#=================================================
# Model fitting
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=12)
classifier.fit(X_train,Y_train)

Y_pred_train = classifier.predict(X_train)
Y_pred_test = classifier.predict(X_test)

classifier.tree_.max_depth
classifier.tree_.node_count

#=================================================

# Tree visualization
# pip install graphviz

from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(classifier,out_file=None, filled=True, rounded=True, special_characters=True)

graph = graphviz.source(dot_data)
graph

#=================================================
# Metrics

from sklearn.metrics import accuracy_score
print("Training accuracy", accuracy_score(Y_train,Y_pred_train).round(3))
print("Testing accuracy", accuracy_score(Y_test,Y_pred_test).round(3))

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_train, Y_pred_train)
cm1
cm2 = confusion_matrix(Y_test, Y_pred_test)
cm2

#================================================
# regularization(Ridge)
from sklearn.linear_model import RidgeClassifier
RC = RidgeClassifier(alpha = 30)
RC.fit(X_train,Y_train)
Y_pred_train = RC.predict(X_train)
Y_pred_test = RC.predict(X_test)


RC.coef_

pd.DataFrame(RC.coef_,axis=1)
pd.DataFrame(X.columns)
d2 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(RC.coef_)],axis = 1)
d2
















