from sklearn import tree
from sklearn import preprocessing
import numpy as np

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
import pydotplus

import pandas

f=pandas.read_csv("Titanic.csv")

label=list(f.columns.values)
survive=f["survived"].values

le = preprocessing.LabelEncoder()
#f=pandas.get_dummies(f)
tset=f[["pclass","sex", "age","sibsp"]].values
for i in range(4):
    tset[:,i] = le.fit_transform(tset[:,i])


X_train, X_test, y_train, y_test = train_test_split(tset, survive, test_size=0.30)



print("full tree")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
ypre=clf.predict(X_test)
print("score: ",clf.score(X_test,y_test))
print(classification_report(y_test, ypre))
first = tree.export_graphviz(clf,rounded=True,filled=True,class_names=["no","yes"],feature_names=["pclass","sex", "age","sibsp"])    
#graph.view("first tree.dot")
#display(SVG(graph.pipe(format='svg')))
graph = pydotplus.graph_from_dot_data(first)  
Image(graph.create_png())

score=1

depth = []

for i in range(3,30):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train,y_train)
    ypre=clf.predict(X_test)
    sc=clf.score(X_test,y_test)
    if score>sc:
        score=sc
        j=i
    print("depth: ",i,"score: ",sc)
    #print(classification_report(y_test, ypre))
    depth.append([i,sc])

a=[]
b=[]
#plt.plot(depth[0],depth[1],"r")
for i in depth:
    a.append(i[0])
    b.append(i[1])


plt.title("figure of depth and F1-score")
plt.plot(a,b,'r')
plt.show()


print('\n\noptimal tree:\n\n')
clf = tree.DecisionTreeClassifier(max_depth=j)
clf = clf.fit(X_train,y_train)
ypre=clf.predict(X_test)
print("depth: ",j,"score: ",clf.score(X_test,y_test))
print(classification_report(y_test, ypre))



a=[]
b=[]
#plt.plot(depth[0],depth[1],"r")
for i in depth:
    a.append(i[0])
    b.append(i[1])



opt=tree.export_graphviz(clf,rounded=True,filled=True,class_names=["no","yes"],feature_names=["pclass","sex", "age","sibsp"])   
#graph.view("optimal tree.dot")
#display(SVG(graph.pipe(format='svg')))
graph = pydotplus.graph_from_dot_data(opt)  
Image(graph.create_png())