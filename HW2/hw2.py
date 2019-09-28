import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.style.use('seaborn-whitegrid') 

def Read():
    fi=pd.read_csv('data2.txt',header=None)
    fi=fi.values
    fi=np.split(fi, 2, axis=1)
    x=fi[0].T
    y=fi[1].T
    x=x.reshape(x.size,1)
    y=y.reshape(y.size,1)
    
    return x,y

def normalEqui(x,y,deg):
    plt.figure(1)
    plt.title("normal equation")
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

def mCo(x,deg):
    xm=np.ones((x.size,1))
    xm=np.append(xm,x,axis=1)#values=np.power(x,f+1)
    return xm



def sto(x,y):
    plt.figure(3)
    plt.title("sto learning rate = 0.02")
    plt.plot(x,y,"ro")
    X=mCo(x,1)
    Y=y
    loop_max = 10000
    
    batch_size = 10
    epsilon = 30
    learning_rate = 0.025
    j=np.zeros([loop_max,4])
    zz=0
    for jj in range(4):
        theta = np.random.rand(2,1)
        for i in range(loop_max):
            idxs = np.random.randint(0,X.shape[0],size=batch_size)
            tmp_X = X.take(idxs,axis=0)
            tmp_y = y.take(idxs,axis=0)
            grad = np.dot(tmp_X.T, (np.dot(tmp_X, theta) - tmp_y))/batch_size
            theta = theta - learning_rate*grad
            j[i][jj] = np.linalg.norm(np.dot(X,theta)-y)
            if j[i][jj] < epsilon:
                print("ok")
                zz=max(zz,i)
                break
        learning_rate=learning_rate-0.00125
    #print(theta)
    if(zz==0):
        zz=loop_max
    plt.plot(x,X.dot(theta),"g")
    return j,zz


def batch(x,y):
    plt.figure(2)
    plt.title("batch  learning rate = 0.02")
    plt.plot(x,y,"ro")
    x_1=mCo(x,1)
    loop_max = 10000
    epsilon = 30
    #theta1 = np.random.rand(2, 1)
    
    y=y.reshape(y.size,1)
    learning_rate = 0.025
    j=np.zeros([loop_max,4])
    zz=loop_max
    for jj in range(4):
        theta=np.zeros([2,1])
        for i in range(loop_max):
            grad = np.dot(x_1.T, (np.dot(x_1, theta) - y)) /y.size
            theta = theta - learning_rate * grad
            j[i][jj] = np.linalg.norm(np.dot(x_1, theta) - y)

            # print("The number of update is %d. The current error is %f"%(i,error))
            if j[i][jj] < epsilon:
                print("ok")
                zz=max(zz,i)
                break
        learning_rate=learning_rate-0.00125
    #print(theta)
    if(zz==0):
        zz=loop_max
    plt.plot(x,x_1.dot(theta),"b")
    return j,zz

def tt(xx):
    x1=xx
    x1.sort()

def main():
    
    x,y=Read()
    normalEqui(x,y,1)#normal equi deg 最高阶
    m1,f1=batch(x,y)
    #m1=bb(x,y,1)
    m2,f2=sto(x,y)
    plt.figure(4)
    plt.title("MSE batch")
    f1=200
    ix=np.arange(0,f1,1)
    plt.axis([0, f1, 0, 200])
    
    for i in range(4):
        y2=m1[:f1,i]
        st='#'+'00'+hex(250-40*i)[2:]+hex(40*i+120)[2:]
        la=str(0.025-0.00125*i)[:7]
        #plt.legend(la)
        plt.plot(ix,y2,label=la,color=st)
    plt.legend()
    plt.figure(5)
    plt.title("MSE sto")
    f2=200
    ix=np.arange(0,f2,1)
    plt.axis([0, f2, 0, 200])
    
    for i in range(4):
        y2=m1[:f2,i]
        st='#'+hex(250-40*i)[2:]+hex(50*i+50)[2:]+'00'
        la='rate= '+str(0.025-0.00125*i)[:7]
        plt.plot(ix,y2,label =la ,color=st)
    plt.legend()
    plt.show()


main() 


