import numpy as np

inputs = np.array([-2, -1, 0, 1, 2])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

print("Inputs")
print(inputs)

print("\nSigmoid")
print(sigmoid(inputs))

print("\nReLU")
print(relu(inputs))
