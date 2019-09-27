In this problem, we write a program to find the coefficients for a linear regression model for the dataset provided (data2.txt). Assume a linear model: y = w0 + w1*x. You need to

1) Plot the data (i.e., x-axis for 1st column, y-axis for 2nd column),


and use Python to implement the following methods to find the 
coefficients:


2) Normal equation, and
3) Gradient Descent using batch AND stochastic modes respectively:\
    a) Determine an appropriate termination condition (e.g., when cost function is less than a threshold, and/or after a given number of iterations).
    b) Print the cost function vs. iterations for each mode; compare and discuss the accuracy of batch and stochastic modes.
    c) Did you see overfitting? If yes, choose one regularization method to control the overfitting. Plot the cost function vs. iteration for each mode again. Discuss how overfitting is controlled if any.
    d) Choose a best learning rate. For example, you can plot cost function vs. learning rate to determine the best learning rate.