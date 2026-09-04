import numpy as np

class LinearRegression:
    #y = b0 + b1X
    def __init__ (self):
        self.b0 = 0
        self.b1 = 0

    def predict(self, X): 
        # y = b0 + b1X
        return self.b0 + self.b1*X
    
    def MSE(self, y, y1):
        # Mean Squared Error
        return np.mean((y-y1)**2)

    def gradients(self,X, y, y1):
        # Derivatives of MSE with respect to b0 and b1
        db0 = -2* np.mean((y-y1))
        db1 = -2* np.mean(X*(y-y1))
        return db0, db1

    def update(self,db0, db1, alpha):
        # Update b0 and b1 using the gradients and learning rate alpha
        self.b0 -= alpha*db0
        self.b1 -= alpha*db1

    def train(self, X, y, alpha, iterations):
        # Train the model using gradient descent
        loss = [] # List to store the loss at each iteration
        for i in range(iterations):
            y1 = self.predict(X)
            error = self.MSE(y, y1)
            db0, db1 = self.gradients(X, y, y1)
            self.update(db0, db1, alpha)
            loss.append(float(error))
        return loss
