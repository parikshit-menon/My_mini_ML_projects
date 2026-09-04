import numpy as np

class LinearRegression:
    def __init__ (self):
        self.b0 = 0
        self.b1 = 0

    def predict(self, X):
        return self.b0 + self.b1*X
    
    def MSE(self, y, y1):
        return np.mean((y-y1)**2)

    def gradients(self,X, y, y1):
        db0 = -2* np.mean((y-y1))
        db1 = -2* np.mean(X*(y-y1))
        return db0, db1

    def update(self,db0, db1, alpha):
        self.b0 -= alpha*db0
        self.b1 -= alpha*db1

    def train(self, X, y, alpha, iterations):
        loss = []
        for i in range(iterations):
            y1 = self.predict(X)
            error = self.MSE(y, y1)
            db0, db1 = self.gradients(X, y, y1)
            self.update(db0, db1, alpha)
            loss.append(float(error))
        return loss
