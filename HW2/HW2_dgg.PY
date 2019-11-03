import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.style.use('seaborn-whitegrid') 

def Read():
    fi=pd.read_csv('data2.txt')
    fi=fi.values
    fi=np.split(fi, 2, axis=1)
    x=fi[0].T
    y=fi[1].T
    x=x.reshape(x.size,1)
    y=y.reshape(y.size,1)
    
    return x,y

def normalEqui(x,y,deg):
    plt.figure(1)
    plt.title("normalequation")
    plt.plot(x,y,"ro")
    l=4
    xn=x.reshape(x.size)
    yn=y.reshape(y.size)
    deg=deg+1
    i=np.identity(deg)
    xm=np.full([x.size,1],1.0)
    for f in range(deg-1):
       xm=np.insert(xm,0,values=np.power(xn,f+1),axis=1)
    w=np.dot(np.linalg.inv(l*i+np.dot(xm.T,xm)),np.dot(xm.T,y))
    i=0
    y1=np.polyval(w,x)
    plt.plot(xn.reshape(xn.size,1),y1,"black")

    return w


def sche(t,t1):
    return 5/(t+t1)/200.0;

def sto(x,y,deg):
    theta=np.random.rand(deg+1,1)
    x2=x
    xm=np.ones((x.size,1))
    xm=np.append(xm,x,axis=1)#
    for epoch in range(50):
        for i in range(y.size):
            r=np.random.randint(y.size)
            xi=np.reshape(x[r],[deg+1,1])
            yi=y[r]
            gradient=2*xi.dot(xi.T.dot(theta)-yi)
            eta=sche(epoch*y.size-i,y.size)
            theta=theta-eta*gradient
    plt.plot(x,np.polyval(theta,x2),"r")
    return theta

def bb(x,y,deg):
    r2=float('inf')
    r1=float('inf')
    x2=x
    x= np.hstack([np.ones((x.size,1)),np.reshape(x,[x.size,1])])
    y=np.reshape(y,[96,1])
    alpha = 0.001 
    theta = np.random.rand(deg+1,1) #初始化参数
    maxCycles = 10000 #迭代次数
    loss=np.zeros(maxCycles)
    for i in range(maxCycles):
        dt=x.T.dot(x.dot(theta)-y)/y.size
        h=x.dot(theta)
        theta=theta-alpha*dt
        loss[i]=np.linalg.norm(y-h)
        if loss[i]<25:
            print(loss[i])
            break
    
    #y=np.polyval(theta,x2)/x2
    plt.plot(x2.T,x.dot(theta),"g")
    plt.show()
    print(theta)
    return theta


def main():
    
    x,y=Read()
    normalEqui(x,y,1)#normal equi deg 最高阶
    m1=bb(x,y,1)
    m2=sto(x,y)
    plt.show()
