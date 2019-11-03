from sklearn import tree
import numpy as np
import graphviz as graph
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import pydotplus

import pandas

f=pandas.read_csv("Titanic.csv")

label=list(f.columns.values)
survive=f["survived"].values

le = preprocessing.LabelEncoder()
tset=f[["pclass","sex", "age","sibsp"]].values
for i in range(4):
    tset[:,i] = le.fit_transform(tset[:,i])


X_train, X_test, y_train, y_test = train_test_split(tset, survive, test_size=0.30)


print("first tree")
clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='random')
clf = clf.fit(X_train,y_train)
ypre=clf.predict(X_test)
print("score: ",clf.score(X_test,y_test))
print(classification_report(y_test, ypre))
first=tree.export_graphviz(clf,rounded=True,filled=True,class_names=["no","yes"],feature_names=["pclass","sex", "age","sibsp"])    
print(first)

score=1


depth = []


for i in range(3,30):
    clf = tree.DecisionTreeClassifier(max_depth=i,criterion='entropy',splitter='random',)
    clf = clf.fit(X_train,y_train)
    ypre=clf.predict(X_test)
    sc=clf.score(X_test,y_test)
    if score>sc:
        score=sc
        j=i
    print("depth: ",i,"score: ",sc)
    depth.append([i,sc])


print('\n\noptimal tree:\n\n')
clf = tree.DecisionTreeClassifier(max_depth=j)
clf = clf.fit(X_train,y_train)
ypre=clf.predict(X_test)
print("depth: ",j,"score: ",clf.score(X_test,y_test))
print(classification_report(y_test, ypre))



opti=tree.export_graphviz(clf,out_file="optimal tree.dot",rounded=True,filled=True,class_names=["no","yes"],feature_names=["pclass","sex", "age","sibsp"])   
#graph.view("optimal tree.dot")
