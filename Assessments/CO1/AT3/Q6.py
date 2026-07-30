#(A)
from sklearn.datasets import make_classification, make_circles
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X1, y1 = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model = Perceptron()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Linearly Separable Accuracy")
print(accuracy_score(y_test, pred))

X2, y2 = make_circles(
    n_samples=300,
    noise=0.1,
    factor=0.4,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nNon-Linearly Separable Accuracy")
print(accuracy_score(y_test, pred))
#(B)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

pred1 = perceptron.predict(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,),
                    max_iter=1000,
                    random_state=42)

mlp.fit(X_train, y_train)

pred2 = mlp.predict(X_test)

print("Perceptron Accuracy:", accuracy_score(y_test, pred1))
print("MLP Accuracy:", accuracy_score(y_test, pred2))
