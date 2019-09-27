# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# Programming Problem:
# 2. [42 points] In this problem, we write a program to estimate the parameters for an unknown polynomial using the polyfit() function of the numpy package.
# 
# 1) Please plot the noisy data and the polynomial you found (in the same figure);
# 2) Change variable noise_scale to 150, 200, 400, 600, 1000 respectively, re-ran the algorithm and plot the polynomials. Discuss the impact of noise scale to the accuracy of the returned parameters.
# 3) Change variable number_of_samples to 40, 30, 20, 10 respectively, re-ran the algorithm and plot the polynomials. Discuss the impact of the number of samples to the accuracy of the returned parameters.
# A simulated dataset will be provided as below. The polynomial used is y = 5 * x + 20 * x2 + x3.
# Simulated data is given as follows in Python: 
#     import matplotlib.pyplot as plt
#     plt.style.use('seaborn-whitegrid') 
#     import numpy as np
#     noise_scale = 100 
#     number_of_samples = 50
#     x = 25*(np.random.rand(number_of_samples, 1) - 0.8)
#     y = 5 * x + 20 * x**2 + 1 * x**3 + noise_scale*np.random.randn(number_of_samples, 1) 
#     plt.plot(x,y,'ro')

#%%
import matplotlib.pyplot as plt
import numpy as np

noise=[100,150, 200, 400, 600, 1000]
sample=[50,40, 30, 20, 10]


def main():
    plt.style.use('seaborn-whitegrid')
    
    for i in noise :
        noise_scale = i 
        for j in sample:
            #x1=[0]*j
            #y1=[0]*j
            number_of_samples = j
            x=25*(np.random.rand(number_of_samples, 1) - 0.8)
            y=5 * x + 20 * x**2 + 1 * x**3 + noise_scale*np.random.randn(number_of_samples, 1) 
            #for k in range(j):
            #    x1[k]=x[k][0]
             #   y1[k]=y[k][0]
            x=np.array(x).flatten()
            y=np.array(y).flatten()
            z=np.polyfit(x,y,5)
            #x1.sort();
            plt.title("noise : "+str(i)+" samples : "+str(j))
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.plot(x,y,'bo')
            x.sort()
            z=np.polyval(z,x)
            plt.plot(x,z,'r')
            plt.show()
    
main()




#%%
