import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op



"""

------------------------------------------------------------------------------------------------------------------------------------------------------

                                                                LINEAR REGRESSION

------------------------------------------------------------------------------------------------------------------------------------------------------


    Clase que permite aplicacion linear regression a un set de datos. Se divide en el set de variables: X, y el set de prediccion: y. La forma de
    matrices con la que se trabajará sera: nxm, donde n son las variables y m cada uno de los ejemplos. De esta manera nuesta matriz X tendra un ejemplo
    por columna y una variable en cada fila.
    

"""

class LinearRegression():

    """

        Inicializamos las variables que utilizaremos en el algoritmo de Linear Regression. Si se indica que axis = 0,
        almacenamos la matriz X de forma normal, si no almacenamos su transpuesta para trabajar siempre con features en
        cada fila y ejemplo en cada columna. Si se indica un split dividimos los datos en el porcentaje indicado.
            
        -- n: numero de variables
        -- m: numero de ejemplos
        -- X: matriz de variables independientes que utilizaremos en el entrenamiento
        -- y: matriz de variable dependiente que utilizaremos en el entrenamiento
        -- X_test: matriz de biases que evaluara el modelo
        -- y_test: matriz de variable dependiente que evaluara el modelo
        -- axis:
                - si 0: las variables están en las filas y los ejemplos en las columnas
                - si 1: las variables están en las columnas y los ejemplos en las filas
        -- split: porcentaje de division de los datos: [0 - 1]
        -- theta: array de biases
    
    """

    def __init__(self, X, y, axis=0, split=None):
        if(axis == 0):
            self.n, self.m = X.shape                                                        # n: features (filas), m: ejemplos (columnas)
            self.X = X                                                                      # Guardamos la matriz de datos original
        else:
            self.m, self.n = X.shape                                                        # n: ejemplos (filas), m: features (columnas)
            self.X = X.T                                                                    # Guardamos la matriz de datos transpuesta: features en filas, ejemplos en columnas

        self.y = y.reshape((1, self.m));                                                    # Obligamos a que y sea una matriz: 1xm
        self.X_test = None                                                                  # Inicializamos X_test
        self.y_test = None                                                                  # Inicializamos y_test

        if split is not None:                                                               # Si se ha introducido algun valor en split
            self._train_test_split(split)                                                   # Separamos los datos

        self.theta = np.matrix(np.zeros(self.n))                                            # Theta: peso de cada variable, sera una matriz 1xn


    """ 
    
        Funcion que separa los datos en train set y test set, se utilizara el primero para entrenar el modelo y el segundo
        para evaluar el rendimiento del mismo.
        
        -- num_train: int que indica el número de ejemplos que se incluiran en el train set.       
         
    """

    def _train_test_split(self, porcentaje):
        num_train = int((1 - porcentaje) * self.m)                                          # Calculamos cuantas columnas seran del training set
        self.X_test = self.X[:, num_train:]                                                 # Obtenemos las columnas hasta el limite
        self.y_test = self.y[:, num_train:]                                                 # Obtenemos las columnas hasta el limite
        self.X = self.X[:, :num_train]                                                      # Obtenemos las columnas restantes
        self.y = self.y[:, :num_train]                                                      # Obtenemos las columnas restantes
        self.n, self.m = self.X.shape                                                       # n: features (filas), m: ejemplos (columnas): actualizamos estos valores al cambiar X e y.


    """ 
    
        Funcion que calcula el coste en funcion del array de biases (peso de cada feature).
        
        -- h: evaluacion de la matriz de datos X
        -- error: diferencia entre la hipotesis y los valores reales
        -- error_sqr: elevar cada elemento al cuadrado
        -- J: suma todos los elementos la matriz hipotesis: matriz fila 1xm
    
    """

    def calcular_coste(self):
        self.theta = self.theta.reshape((1, self.n));                                       # Hacemos que theta sea un vector fila: 1 x n
        h = self.theta * self.X                                                             # Evaluamos la funcion con todos los datos de entrada
        error = h - self.y                                                                  # Obtenemos el error
        error_sqr = np.power(error, 2)                                                      # Elevamos el error al cuadrado
        J = np.sum(error_sqr) / (2 * self.m)                                                # Hacemos el sumatorio y lo dividimos por el doble del numero de muestras
        return J                                                                            # Devolvemos el resultado


    """
    
        Funcion que calcula el coste en funcion del array de biases (peso de cada feature) utilizada en la funcion de minimizacion.
        
        -- h: evaluacion de la matriz de datos X
        -- error: diferencia entre la hipotesis y los valores reales
        -- error_sqr: elevar cada elemento al cuadrado
        -- J: suma todos los elementos la matriz hipotesis: matriz fila 1xm
    
    """

    def _calcular_coste_min(self, theta, X, y):
        theta = theta.reshape((1, self.n));                                                 # Hacemos que theta sea un vector fila: 1 x n
        h = theta * X                                                                       # Evaluamos la funcion con todos los datos de entrada
        error = h - y                                                                       # Obtenemos el error
        error_sqr = np.power(error, 2)                                                      # Elevamos el error al cuadrado
        J = np.sum(error_sqr) / (2 * self.m)                                                # Hacemos el sumatorio y lo dividimos por el doble del numero de muestras
        return J                                                                            # Devolvemos el resultado


    """ 
    
        Funcion que calcula el gradiente de cada uno de los biases en una matriz fila: 1xn.
        
        -- h: evaluacion de la matriz de datos X
        -- error: diferencia entre la hipotesis y los valores reales
        -- gradiente: multiplicar cada fila de error (todos los ejemplos), con cada columna de la transpuesta de la matriz de datos X
    
    """

    def _gradiente(self):
        self.theta = self.theta.reshape((1, self.n));                                       # Hacemos que theta sea un vector fila: 1 x n
        h = self.theta * self.X                                                             # Evaluamos la funcion con todos los datos de entrada
        error = h - self.y                                                                  # Calculamos el error
        gradiente = (error * self.X.T) / self.m                                             # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        return np.ravel(gradiente)                                                          # Hacemos que gradiente sea un vector fila: 1 x n


    """ 

        Funcion que calcula el gradiente de cada uno de los biases en una matriz fila: 1xn para la funcion de minimizacion que 
        requiere la introduccion de argumentos.

        -- h: evaluacion de la matriz de datos X
        -- error: diferencia entre la hipotesis y los valores reales
        -- gradiente: multiplicar cada fila de error (todos los ejemplos), con cada columna de la transpuesta de la matriz de datos X

    """

    def _gradiente_min(self, theta, X, y):
        theta = theta.reshape((1, self.n));                                                 # Hacemos que theta sea un vector fila: 1 x n
        h = theta * X                                                                       # Evaluamos la funcion con todos los datos de entrada
        error = h - y                                                                       # Calculamos el error
        gradiente = (error * X.T) / self.m                                                  # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        return np.ravel(gradiente)                                                          # Hacemos que gradiente sea un vector fila: 1 x n


    """ 
    
    
        Funcion que lleva a cabo el algoritmo de gradient descent para hayar el coste minimo. En cada iteraccion comprobamos que con la nueva 
        theta el valor del coste no ha aumentado, en caso contrario se acaba el algoritmo.
        
        -- coste_anterior: coste obtenido en la ronda anterior a la actual
        -- coste_actual: coste obtenido en la ronda actual
        -- theta_anterior: theta obtenido en la ronda anterior
        -- step: learning rate
        -- iter: numero de iteracciones
    
    """

    def gradient_descent(self, step, iter):
        coste_anterior = self.calcular_coste()                                              # Creamos la variable que contendra el coste de la ronda anterior
        theta_anterior = self.theta                                                         # Guardamos el valor del peso de las variables de la ronda anterior
        for i in range(iter):
            self.theta = self.theta - step * self._gradiente()                              # Calculamos la nueva theta
            coste_actual = self.calcular_coste()                                            # Calculamos el nuevo coste
            if (coste_actual > coste_anterior):                                             # Si el nuevo coste es mayor, paramos
                self.theta = theta_anterior                                                 # Nos quedamos con la theta de la ronda anterior
                break
            else:
                coste_anterior = coste_actual                                               # Actualizamos el coste de la ronda anterior como preparacion para la siguiente ronda
                theta_anterior = self.theta                                                 # Actualizamos la theta de la ronda anterior


    """ 
    
        Funcion que minimiza el coste utilizando la optimizacion de scipy.
        
        -- initial_theta: array de biases inicializado a cero
        -- Result: resultado de la minimizacion
    
    """

    def minimizacion(self):
        initial_theta = np.zeros(self.n);                                                   # Inicializamos un array de variables auxiliares a cero

        Result = op.minimize(fun=self._calcular_coste_min,                                  # Optimizamos la funcion de coste
                             x0=initial_theta,
                             args=(self.X, self.y),
                             method='TNC',
                             jac=self._gradiente_min);

        self.theta = Result.x;                                                              # Actualizamos el array de variables


    """ 
    
        Funcion que calcula la prediccion dada unos datos de entrada.
    
    """

    def prediccion(self, test):
        return np.ravel(self.theta) * test


    """ 

        Funcion que calcula la prediccion de los datos test, en caso de haber realizado un split en la inicializacion del modelo.

    """

    def prediccion_test(self):
        if self.X_test is not None:
            return np.ravel(self.theta) * self.X_test


    """ 
    
        Funcion que crea la grafica de los datos con la representación del modelo.
    
    """

    def plot_regression(self, titulo, xlabel, ylabel):
        plt.scatter(x=np.ravel(self.X[1, :]), y=np.ravel(self.y[0, :]), color='red')        # Scatter plot de los datos
        plt.title(titulo)                                                                   # Titulo de la grafica
        plt.xlabel(xlabel)                                                                  # Etiqueta de las x
        plt.ylabel(ylabel)                                                                  # Etiqueta de las y
        plt.plot(np.ravel(self.X[1, :]), np.ravel(self.prediccion(self.X)))                 # Grafica de preddicion: linear
        plt.show()                                                                          # Mostrar grafica


    """ 

        Funcion que crea la grafica de los datos con la representación del modelo con los datos de test.

    """

    def plot_regression_test(self, titulo, xlabel, ylabel):
        if self.X_test is not None:
            plt.scatter(x=np.ravel(self.X_test[1, :]), y=np.ravel(self.y_test[0, :]), color='red')  # Scatter plot de los datos
            plt.title(titulo)                                                                       # Titulo de la grafica
            plt.xlabel(xlabel)                                                                      # Etiqueta de las x
            plt.ylabel(ylabel)                                                                      # Etiqueta de las y
            plt.plot(np.ravel(self.X_test[1, :]), np.ravel(self.prediccion_test()))                 # Grafica de preddicion: linear
            plt.show()                                                                              # Mostrar grafica


    """ 

        Funcion que aplica feature normalization a los datos, pudiendo elegir si se aplica los datos y a las predicciones.
        Tambien se permite excluir algunos de los biases en caso de ser one-hot-encoded. Por defecto no se incluye la primera
        fila ya que se corresponde con el termino independiente de los biases: x0.
        
        -- X_norm: matriz de datos normalizada que no incluye la primera fila de unos.
        -- exclude: rango de indice excluidos
        -- start: indice de comienzo de exclusion
        -- stop: indice de final de exclusion
        -- exclude_X: matriz que guarda la submatriz de datos que no queremos normalizar

    """

    def aplicar_feature_normalization(self, exclude=None, include_y=False):
        if exclude is None:
            X_norm = self._feature_normalization(self.X[1:])                        # Matriz normalizada que no incluye la fila de 1 correspondientes con x0
            self.X = np.insert(self.X[0], 1, X_norm, axis=0)                        # Concatenamos los 1 de la primera fila con la matriz normalizada
        else:
            start = exclude[0]                                                      # Indice de comienzo
            stop = exclude[1] + 1                                                   # Indice de fin
            exclude_X = self.X[start:stop, :]                                       # Matriz con los elementos excluidos de la normalizacion
            self.X = np.delete(self.X, [range(start, stop)], axis=0)                # Eliminar elementos que no queremos normalizar
            X_norm = self._feature_normalization(self.X[1:])                        # Matriz normalizada
            self.X = np.insert(self.X[0], 1, X_norm, axis=0)                        # Concatenamos los 1 de la primera fila con la matriz normalizada
            self.X = np.concatenate((self.X, exclude_X), axis=0)                    # Concatenamos el resto de elementos excluidos en la matriz

        if include_y:
            self.y = self._feature_normalization(self.y)                            # Normalizamos los targets


    """ 
    
        Funcion que calcula la normalizacion para cada uno de los elementos de la matriz de entrada.
        
        -- media: media de cada fila: todos los ejemplos de cada feature
        -- std: desviacio tipica de cada fila: todos los ejemplos de cada feature
        -- sub_media: la matriz de entrada menos la media
        -- normalizado: sub_media dividido por la desviacion típica
    
    """

    def _feature_normalization(self, X):
        media = np.mean(X, axis=1)                                                          # Calculamos la media de cada fila, axis = 1
        std = np.std(X, axis=1)                                                             # Calculamos la desviacion tipica de cada columna
        sub_media = X - media                                                               # Restamos la media de cada columna a cada elemento de dicha columna
        normalizado = sub_media / std                                                       # Dividimos cada elemento de una columna por su desviacion tipica
        return normalizado                                                                  # Devolvemos el set normalizado


    """ 
        
        Funcion que utiliza la ecuacion normal para calcular la matriz de biases que supone el coste minimo.
        
        -- X: matriz de datos transpuesta, debe ser asi para realizar los calculos correctamente
        -- y: matriz de targets transpuesta, idem.

    """

    def norm_equation(self):
        X = self.X.T
        y = self.y.T
        self.theta = np.linalg.inv(X.T * X) * (X.T * y)                                     #Calculamos el mínimo con la funcion de la ecuacion normal:
                                                                                            #theta = ((X*X_transpuesta)^inversa)*(X_transpuesta*y)