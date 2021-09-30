import numpy as np
import scipy.optimize as op

"""

------------------------------------------------------------------------------------------------------------------------------------------------------

                                                                SVM

------------------------------------------------------------------------------------------------------------------------------------------------------



"""

class SVM:
    
    def __init__(self, X, y, axis=0, C=1, lr=0.01):
        if axis == 0:
            self.X = X
        else:
            self.X = X.T
            
        self.n, self.m = self.X.shape
        y[y == 0] = -1
        self.y = np.reshape(y, (1, self.m))
        self.X = np.row_stack((np.ones(self.m), self.X))
        self.n, self.m = self.X.shape
        self.C = C 
        self.lr = lr
        self.theta = np.ones((1, self.n))
        
    def calcular_distancias(self, theta=None):
        
        if theta is None:
            theta = self.theta
            
        distancias = np.multiply(self.y, (np.dot(theta, self.X))) - 1
        return distancias
    
    def gradiente(self, theta=None):
        
        if theta is None:
            theta = self.theta
        
        
        distancias = self.calcular_distancias(theta)

        L = 1 / 2 * np.sum(np.power(theta, 2)) - self.C * np.sum(distancias)

        dw = np.zeros((1, self.n))

        for ind, d in enumerate(np.ravel(distancias)):
            if d == 0:  
                di = theta  
            else:
                di = theta - (self.C * self.y[0, ind] * self.X[:, ind])
            dw += di
        
        return L, np.ravel(dw / self.m)
    
    def fit(self, iter=100):
        for i in range(iter):
            L, dw = self.gradiente()
            self.theta = self.theta - self.lr * dw
            
    def minimizar(self):
        theta_init = np.ones((1, self.n))
        
        Result = op.minimize(fun=self.gradiente,           # Funcion a minimizar
                                 x0=theta_init,            # Primer argumento
                                 method='L-BFGS-B',
                                 jac=True);
        self.theta = Result.x
        self.theta = np.reshape(self.theta, (1, self.n))
        

    def predict(self, X=None):
        if X is None:
            return np.sign(self.theta @ self.X)
        else:
            n, m = X.shape
            X = np.row_stack((np.ones(m), X))
            return np.sign(self.theta @ X)
    
    def accuracy(self):
        return np.sum((self.y == self.predict())*1)/self.m