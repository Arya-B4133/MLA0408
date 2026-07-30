#A
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)
y = 3*x + 2

w = 0
b = 0

lr = 0.01

losses = []

for i in range(500):

    pred = w*x + b

    loss = np.mean((y-pred)**2)
    losses.append(loss)

    dw = (-2/len(x))*np.sum(x*(y-pred))
    db = (-2/len(x))*np.sum(y-pred)

    w -= lr*dw
    b -= lr*db

print("Weight:", round(w,3))
print("Bias:", round(b,3))

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent")
plt.show()
#B
import numpy as np
from sklearn.linear_model import SGDRegressor

X = np.arange(1,21).reshape(-1,1)
y = 3*X.ravel() + 2

model = SGDRegressor(max_iter=1000,
                     learning_rate='constant',
                     eta0=0.01,
                     random_state=42)

model.fit(X,y)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_[0])
