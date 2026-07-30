import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

m = 0
b = 0
lr = 0.01
epochs = 1000

n = len(x)
losses = []

for i in range(epochs):
    y_pred = m * x + b

    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m -= lr * dm
    b -= lr * db

print("Slope:", round(m,4))
print("Intercept:", round(b,4))
print("Final Loss:", round(losses[-1],6))

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()
