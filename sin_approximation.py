import numpy as np
import matplotlib.pyplot as plt
from activation import Activation

# 1. Generate Data
np.random.seed(42)
x = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

# network setting
input_size = 1
hidden_size = 16
output_size = 1

# parameter initialization
learning_rate = 0.1
epochs = 1000
a= Activation()

# weights initialization
w1 = np.random.randn(input_size,hidden_size)
b1 = np.zeros((1,hidden_size))
w2 = np.random.randn(hidden_size,output_size)
b2 = np.zeros((1,output_size))

# # Activation functions
# class Activation:
#   def tanh(self,z):
#     return np.tanh(z)
#   def tanh_derivative(self,z):
#     return 1 - np.tanh(z)**2
#   def sigmoid(self,x):
#     return 1 / (1 + np.exp(-x))
#   def sigmoid_derivative(self,x):
#     return Activation().sigmoid(x) * (1-Activation().sigmoid(x))
  
# loss function initialization 
class Loss:
  def loss_function(self,y_true,y_pred):
   
    return sum((y-y_pred)**2)/y.shape[0]

for epoch in range(epochs):
  # forward pass
  
  z1 = x @ w1 + b1
  a1 = a.sigmoid(z1)
  z2 = a1 @ w2 + b2
  y_pred  = z2
  
  loss = Loss().loss_function(y,y_pred)
  # backpropagation
  dy_pred = -2*(y-y_pred)/y.shape[0]
  dz2 = dy_pred
  dw2 = a1.T @ dz2
  db2 = np.sum(dy_pred, axis=0, keepdims=True)
  da1 = dz2 @ w2.T
  dz1 = da1 * a.sigmoid_derivative(z1)
  dw1 = x.T @ dz1
  db1 =  np.sum(dz1, axis=0, keepdims=True)

  # now upgrade the parameters by gradient descent method
  w2 = w2 - learning_rate*dw2
  w1 = w1 - learning_rate*dw1
  b2 = b2 - learning_rate*db2
  b1 = b1 - learning_rate*db1

# 4. Plot Results
plt.plot(x, y, label='True sin(x)')
plt.plot(x, y_pred, label='NN Approximation')
plt.legend()
plt.title("Approximating sin(x) using a neural network from scratch")
plt.show()
# R² score
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2_score = 1 - (ss_res / ss_tot)

print(f"R² Score = {r2_score:.6f}")