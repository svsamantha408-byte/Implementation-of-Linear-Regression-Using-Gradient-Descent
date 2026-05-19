# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a dataset with input values (X) and output values (y).

2. Initialize the slope (m), intercept (b), learning rate, and number of epochs.

3. Apply Gradient Descent to calculate prediction errors and compute gradients.

4. Update the slope and intercept repeatedly to reduce the error.

5. Use the final values of m and b to predict results and plot the regression line.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (X = input, y = output)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Initialize parameters
m = 0  # slope
b = 0  # intercept

# Hyperparameters
learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Compute gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Final parameters
print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions
y_pred = m * X + b

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()



```

## Output:
![linear regression using gradient descent](sam.png)

<img width="1017" height="617" alt="Screenshot 2026-05-19 153305" src="https://github.com/user-attachments/assets/1a7dc367-72dc-471c-8c62-47cd7811d2c7" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
