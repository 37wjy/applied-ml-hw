import matplotlib.pyplot as plt
import numpy as np


# Read data and plot the data
with open('data2.txt', 'r') as f:
    data = f.readlines()
xlist = []
ylist = []
for i in data:
    xlist.append(float(i.split(",")[0]))
    ylist.append(float(i.split(",")[1]))
x = np.array(xlist)
y = np.array(ylist)
plt.figure("test")
plt.scatter(x, y, color = 'red', label = 'Data')

#Normal Equation
x_bias = np.ones((len(x),1))
x_1 = np.reshape(x,(len(x),1))
x_1 = np.append(x_bias,x_1,axis=1)
x_transpose = x_1.T
x_transpose_dot_x = x_transpose.dot(x_1)
temp_1 = np.linalg.inv(x_transpose_dot_x)
temp_2=x_transpose.dot(y)
theta =temp_1.dot(temp_2)
print(theta)
plt.plot(x, theta[0]+theta[1]*x, color = 'blue', label ='Normal Equation')


#Batch
loop_max = 10000
epsilon = 25
theta1 = np.random.rand(2, 1)
learning_rate = 0.001
for i in range(loop_max):
    grad = np.dot(x_1.T, (np.dot(x_1, theta1) - y.reshape(len(y),1))) / x_1.shape[0]
    theta1 = theta1 - learning_rate * grad
    error = np.linalg.norm(np.dot(x_1, theta1) - y)
    # print("The number of update is %d. The current error is %f"%(i,error))
    if error < epsilon:
        break

print(theta1.reshape(1,2))
plt.plot(x, theta1[0] + theta1[1]*x, color='green', label = 'Batch')


plt.legend()
plt.show()