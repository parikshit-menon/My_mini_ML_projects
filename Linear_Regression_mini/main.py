import pandas as pd
import matplotlib.pyplot as plt
from src.Linear_R import LinearRegression

data = pd.read_csv("data/houses.csv")

print(data, "\n")


X = data["size"].to_numpy()
y = data["price"].to_numpy()

plt.scatter(X, y)
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Prices vs Size")

model = LinearRegression()

loss = model.train(X, y, 0.01, 100)
plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Training Loss")
plt.show()
print("b0:", model.b0)
print("b1:", model.b1)