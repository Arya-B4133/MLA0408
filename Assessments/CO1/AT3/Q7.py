#A
import numpy as np

inputs = np.array([1, 2])

weights_input_hidden = np.array([[0.2, 0.4],
                                 [0.3, 0.5]])

weights_hidden_output = np.array([[0.6],
                                  [0.7]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

hidden_input = np.dot(inputs, weights_input_hidden)
hidden_output = sigmoid(hidden_input)

final_input = np.dot(hidden_output, weights_hidden_output)
final_output = sigmoid(final_input)

print("Hidden Layer Output:")
print(hidden_output)

print("\nFinal Output:")
print(final_output)
#B
import numpy as np

x = np.array([[1, 0]])
y = np.array([[1]])

w = np.random.rand(2,1)

lr = 0.1

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return z*(1-z)

output = sigmoid(np.dot(x,w))

error = y - output

gradient = error * sigmoid_derivative(output)

w = w + lr * np.dot(x.T, gradient)

print("Updated Weights:")
print(w)
