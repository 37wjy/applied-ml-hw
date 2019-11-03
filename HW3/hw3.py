
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing

#from IPython.display import SVG
#from graphviz import Source
#from IPython.display import display
from IPython.display import Image

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data as skshow


f=pd.read_csv("Titanic.csv")

label=list(f.columns.values)
survive=f["survived"].values

le = preprocessing.LabelEncoder()
#f=pandas.get_dummies(f)
tset=f[["pclass","sex", "age","sibsp"]].values
for i in range(4):
    tset[:,i] = le.fit_transform(tset[:,i])


X_train, X_test, y_train, y_test = train_test_split(tset, survive, test_size=0.30)

#d=graph.Digraph()

print("first tree")
clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='random')
clf = clf.fit(X_train,y_train)
ypre=clf.predict(X_test)
print("score: ",clf.score(X_test,y_test))
print(classification_report(y_test, ypre))
first=tree.export_graphviz(clf,rounded=True,filled=True,class_names=["no","yes"],feature_names=["pclass","sex", "age","sibsp"])    
#graph.view("first tree.dot")
first=skshow(first)
Image(first.creat)


score=0
dp=clf.get_depth()
depth = []

for i in range(2,dp):
    f1=0
    f2=0
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train,y_train)
    ypre=clf.predict(X_test)
    typre=clf.predict(X_train)
    sc=clf.score(X_test,y_test)
    if score<sc:
        score=sc
        j=i
    for c in range(0,len(typre)):
        if typre[i] != y_train :
            f1=f1+1
        if ypre[c] != y_test :
            f2=f2+1
    print("depth: ",i,"score: ",sc)
    #print(classification_report(y_test, ypre))
    depth.append([i,sc,f1,f2])


a=[]
b=[]
f1=[]
f2=[]
#plt.plot(depth[0],depth[1],"r")
for i in depth:
    a.append(i[0])
    b.append(i[1])
    f1.append(i[2])
    f2.append(i[3])

plt.title("figure of depth and F1-score")
plt.plot(a,b,'r')
plt.show()
plt.title("misclassified node(in sample)")
plt.plot(a,f1,'r')
plt.show()
plt.title("misclassified node(out of sample)")
plt.plot(a,f2,'r')
plt.show()



print('\n\noptimal tree:\n\n')
clf = tree.DecisionTreeClassifier(max_depth=j)
clf = clf.fit(X_train,y_train)
ypre=clf.predict(X_test)
print("depth: ",j,"\nscore(out of sample): ",clf.score(X_test,y_test))
print(classification_report(y_test, ypre))
ypre=clf.predict(X_train)
print("scor(in sample)e: ",clf.score(X_train,y_train))
print(classification_report(y_train, ypre))


opti=tree.export_graphviz(clf,out_file="optimal tree.dot",rounded=True,filled=True,class_names=["no","yes"],feature_names=["pclass","sex", "age","sibsp"])   
#graph.view("optimal tree.dot")
graph = pydotplus.graph_from_dot_data(poti)  
Image(graph.create_png())