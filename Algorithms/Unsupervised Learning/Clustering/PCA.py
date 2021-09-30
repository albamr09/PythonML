import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos de PCA, es decir, algoritmos de reduccion de dimension.
            
            -- X: matriz de terminos independientes.
            -- n: numero de features, numero de columnas.
            -- m: numero de ejemplos, numero de filas.
            -- media: lista de medias de cada feature.
            -- std: lista de desviacion tipica de cada feature.
            -- X_approx: proyeccion de X sobre el "plano" de menor dimension.
            -- u_reduce: matriz de eigenvectors que se utilizará para reducir la dimensión de los datos.
            -- z: "plano" de menor dimension, nuevas features.

------------------------------------------------------------------------------------------------------------------------

"""


class PCA:
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: si 0 -> features en columnas y ejemplos en filas, si no viceversa.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def __init__(self, X, axis=0):
        
        if axis == 0:
            self.X = X                      # Inicializamos X de forma normal
        else:
            self.X = X.T                    # Almacenamos la transpuesta de X para que las filas sean ejemplo y las columnas features.
        
        self.m, self.n = self.X.shape       # Guardamos el numero de ejemplos (m) y el número de de features (n)
        
        self.data_preprocessing()           # Aplicamos feature normalization
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la media de cada feature.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _mean_normalization(self):
        self.media = np.mean(self.X, axis=0)    # Calculamos y almacenamos la media de cada feature calculando la media de cada columna
        self.X -= self.media                    # Se lo restamos a X

        
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la desviación típica de cada feature.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _feature_scaling(self):
        self.std = np.std(self.X, axis=0)   # Calculamos y almacenamos la desviación típica de cada feature calculando la desviación típica de cada columna
        self.X /= self.std                  # Dividimos X entre el valor calculado en cada columna
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que aplica feature normalization.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def data_preprocessing(self):
        self._mean_normalization()          # Restamos la media
        self._feature_scaling()             # Dividimos la desviación típica
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion calcula la matriz de covarianza

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _matriz_covarianza(self):
        return (1/self.m) * self.X.T.dot(self.X)        # Aplicamos la formula: 1/m * (X.T*X) 
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion proyecta los datos originales sobre los de menor dimension.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _reconstruccion(self):
        self.X_approx = self.u_reduce.dot(self.z).T     # Aplicamos la fórmula: (U_reduce * z).T -> Ejemplos en filas, features en columnas
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion lleva a cabo el algoritmo de PCA:
            
            1. Calculamos la matriz de covarianza.
            2. Calculamos los eigenvectors.
            3. Proyectamos X para hayar z.
            
            -- nueva_dim: dimension a la que vamos a reducir los features.
            -- visualizar: variable de control que permite visualizar los eigenvectors.
            -- proyeccion: variable de control que permite visualizar la proyección de X sobre el "plano" z

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def aplicar_PCA(self, nueva_dim=None, visualizar=False, proyeccion=False, limite_varianza=0.01):
        
        sigma = self._matriz_covarianza()                                   # Calculamos la matriz de covarianza     
        u, s, v = np.linalg.svd(sigma)                                      # Calculamos los eigenvectors
        
        if nueva_dim is None:                                               # Si no se pasa una dimension por argumento se calcula la mas optima
            dim = 1
            while True:
                varianza = 1- (np.sum(s[:dim]) / np.sum(s))                 # Calculamos la relacion de varianza
                if varianza <= limite_varianza:                             # Si se cumple con la condicion de limite de varianza
                    nueva_dim = dim                                         # Se guarda la dimension
                    break
                else:                                                       # Si no calculamos la siguiente
                    dim += 1
                    
        if visualizar:
            plt.title("EigenVectors")
            plt.plot(self.media, self.media + 1.5 * s[0] * u[:, 0])         # Calcula los puntos
            plt.plot(self.media, self.media + 1.5 * s[1] * u[:, 1])         # Calcula los puntos
            plt.show()
        
        self.u_reduce = np.reshape(u[:, :nueva_dim], (self.n, nueva_dim))   # Cogemos solo las columnas hasta la nueva dimension indicada
        self.z = self.u_reduce.T.dot(self.X.T)                              # Calculamos la proyección z
        self._reconstruccion()                                              # Calculamos la aproximación de X a través de z y los eigenvectors
        
        if proyeccion:
            plt.title("Proyeccion")
            plt.scatter(self.X_approx[:, 0], self.X_approx[:, 1],           # Hacer plot de la proyección de z en 2D
                        facecolors='none', 
                        edgecolors='r', 
                        label="Proyeccion")
            
            plt.scatter(self.X[:, 0], self.X[:, 1],                         # Hacer plot de los datos originales
                        facecolors='none', 
                        edgecolors='blue', 
                        label="Original")
            
            for i in range(self.m):
                plt.plot([self.X[i, 0], self.X_approx[i, 0]],               # Hacer plot de lineas desde X hasta la aproximación de X
                         [self.X[i, 1], self.X_approx[i, 1]], 
                         'k--')
                
            plt.legend()
            plt.show()
            
        
        
