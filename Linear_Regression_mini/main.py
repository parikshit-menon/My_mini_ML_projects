import pandas as pd
import numpy as np
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
# Train the model with learning rate 0.01 and 100 iterations
# Plot the training loss over iterations
plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Training Loss")
plt.show()

# the new values of b0 and b1 after training
print("b0:", model.b0)
print("b1:", model.b1)

# Make predictions for new data points
new_sizes = np.array([2.5, 3.5])
new_predictions = model.predict(new_sizes)
print("Predictions:", new_predictions)

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red') # Plot the regression line
plt.plot(new_sizes, new_predictions, 'go') # Plot the new predictions
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Prices vs Size with Regression Line")
plt.show()

# Evaluate the model on the training data
final_predictions = model.predict(X)
final_mse = model.MSE(y, final_predictions)
print("Final MSE:", final_mse)