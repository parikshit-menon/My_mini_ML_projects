import numpy as np

class LinearRegression:
    #y = b0 + b1X
    def __init__ (self):
        self.theta = None

    '''def predict(self, X): 
        # y = b0 + b1X
        return self.b0 + self.b1*X'''
    def predict(self, X):
        X = np.asarray(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta
    
    def MSE(self, y, y1):
        # Mean Squared Error
        return np.mean((y-y1)**2)

    '''def gradients(self,X, y, y1):
        # Derivatives of MSE with respect to b0 and b1
        db0 = -2* np.mean((y-y1))
        db1 = -2* np.mean(X*(y-y1))
        return db0, db1

    def update(self,db0, db1, alpha):
        # Update b0 and b1 using the gradients and learning rate alpha
        self.b0 -= alpha*db0
        self.b1 -= alpha*db1'''

    def gradients(self, X, y, y_pred):
        error = y_pred - y
        return (2 / len(y)) * (X.T @ error)

    def update(self, gradient, alpha):
        self.theta -= alpha * gradient

    def train(self, X, y, alpha, iterations):
        X = np.asarray(X)
        X = np.c_[np.ones(X.shape[0]), X]

        self.theta = np.zeros(X.shape[1])
        # Train the model using gradient descent
        loss = [] # List to store the loss at each iteration
        for i in range(iterations):
            y1 = X @ self.theta
            error = self.MSE(y, y1)
            gradient = self.gradients(X, y, y1)
            self.update(gradient, alpha)
            loss.append(float(error))
        return loss
    
    def normal_equation_closed(self, X, y):
        X = np.c_[np.ones(len(X)), X]
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y #this computes the closed form equation - (X^T X)^{-1}X^T y
        #X.T= the transpose of X 
        #.inv(A) computes the inverse o a matrix A
        # the @ is matrix multiplication
        self.b0 = self.theta[0]
        self.b1 = self.theta[1]

        return self.theta

    def normal_equation(self, X, y):
        X = np.asarray(X)
        X = np.c_[np.ones(X.shape[0]), X]

        self.theta = np.linalg.pinv(X) @ y
        return self.theta