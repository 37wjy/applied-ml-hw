import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def Read():
    fi=pd.read_csv('data2.txt')
    fi=fi.values
    fi=np.split(fi, 2, axis=1)
   # plt.show()
    x=fi[0].T
    y=fi[1].T
    #plt.plot(x,y,"ro")
    return x,y

def logit(x):
    y=np.zeros(x.shape)
    y=1./(1+np.exp(-x))
    return y

def normalEqui(x,y,deg):
    pt=plt
    l=4
    deg=deg+1
    plt.plot(x,y,"ro")
    i=np.identity(deg)
    xm=np.full([x.size,1],1.0)
    for f in range(deg-1):
       # print(pow(x,f))
       xm=np.insert(xm,0,values=np.power(x,f+1),axis=1)
    w=np.dot(np.linalg.inv(l*i+np.dot(xm.T,xm)),np.dot(xm.T,y))
    i=0
    x.sort()
    x2=np.arange(x[0],x[-1],(x[-1]-x[0])/x.size)
    plt.plot(x2,np.polyval(w,x2),"b")
    return w

def mCo(x,deg):
    xm=np.ones((x.size,1))
    x=x.reshape(x.size,1)
    #for f in range(deg-1):
     #   print(pow(x,f))
    xm=np.append(xm,x,axis=1)#values=np.power(x,f+1)
    return xm

""" def MSE(x,y,theta,deg):

    e=np.sum(pow(np.dot(theta.T,x)-y,2)/x.shape[1])
    return np.sum(np.dot(np.dot(theta,x.T)-y,x)),e """

def batch(x,y):
    x_1=mCo(x,1)
    loop_max = 10000
    epsilon = 25
    theta1 = np.random.rand(2, 1)
    y=y.reshape(y.size,1)
    learning_rate = 0.001
    for i in range(loop_max):
        grad = np.dot(x_1.T, (np.dot(x_1, theta1) - y)) / 96
        theta1 = theta1 - learning_rate * grad
        error = np.linalg.norm(np.dot(x_1, theta1) - y)
        # print("The number of update is %d. The current error is %f"%(i,error))
        if error < epsilon:
            break
    print(theta1)
    x.sort()
    plt.plot(x,theta1[0]+theta1[1]*x,"b")


def bb(x,y,deg):
    r2=float('inf')
    r1=float('inf')
    x2=x
    x=mCo(x,deg+1)

    y=np.reshape(y,[96,1])
    alpha = 0.001 
    theta_g = np.random.rand(deg+1,1) #初始化参数
    theta_x=np.zeros(1000)
    theta_y=np.zeros(1000)
    maxCycles = 10000 #迭代次数
    loss=np.zeros(maxCycles)
    for i in range(maxCycles):

        dt=x.T.dot(x.dot(theta_g)-y)/y.size

        h=x.dot(theta_g)
        err=np.linalg.norm(y-h)
        theta_g=theta_g-alpha*dt
        
        if err<25:
            break
        #theta_x[i]=theta_g[0]
        #theta_y[i]=theta_g[1]
    #plt.plot(x2,np.polyval(theta_g,x2),"g")
    plt.plot(x2,theta_g[0]+theta_g[1]*x2,"g")
    #plt.plot(theta_x,theta_y,"g")
    #plt.plot(theta_x[-1],theta_y[-1],"ro")
    print(theta_g)
    return theta_g

def sche(t,t1):
    return 5/(t+t1)/200.0;

def sto(x,y,deg):
    theta=np.random.rand(deg+1,1)
    x2=x

    x=mCo(x,deg+1)
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
    

    return 0

def main():
    x,y=Read()
    #normalEqui(x,y,1)#normal equi deg 最高阶
    #batch(x,y)
    #plt.plot(x,y,"ro")
    z=bb(x,y,1)
    #sto(x,y,1)
    plt.show()
main()