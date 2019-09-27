import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid') 
import numpy as np
noise_scale = 100 
number_of_samples = 50
x = 25*(np.random.rand(number_of_samples, 1) - 0.8)
y = 5 * x + 20 * x**2 + 1 * x**3 + noise_scale*np.random.randn(number_of_samples, 1) 

plt.plot(x,y,'ro')
plt.show()
