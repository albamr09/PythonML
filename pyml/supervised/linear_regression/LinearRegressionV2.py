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
        -- split: porcentaje de division de los datos en train set, test set: [0 - 1]
        -- theta: array de biases
        -- test: indica si ya hay una entrada para el test set
        -- val: indica si ya hay una entrada para el validation set
        -- val-split: porcentaje de division de los datos en train set y validation set: [0 - 1]
        -- reg: indica si se aplicara regularizacion
        -- reg_par: parametro de regularizacion
        -- poly: indica si se utilizara polynomial regression
        -- grado: grado en cual aumentaremos la complejidad del polinomio
        -- feature_norm: indica si se realizara feature normalization
    
    """

    def __init__(self, X, y, axis=0, split=None, test=None, val=None, val_split=None, reg=False, reg_par=None, grado=None, feature_norm=False):
        if(axis == 0):        
            self.X = X                                                                      # Guardamos la matriz de datos original
        else:
            self.X = X.T                                                                    # Guardamos la matriz de datos transpuesta: features en filas, ejemplos en columnas

        
        self.n, self.m = self.X.shape                                                       # n: features (filas), m: ejemplos (columnas)
        self.X = np.concatenate((np.matrix(np.ones(self.m)), self.X))                       # Añadimos unos al término independiente
        self.y = y.reshape((1, self.m));                                                    # Obligamos a que y sea una matriz: 1xm
        self.n, self.m = self.X.shape                                                       # Actualizamos n: features (filas), m: ejemplos (columnas)
        
        if grado is not None:                                                               # Si se ha indicado un grado para aumentar features
            self.grado = grado                                                              # Guardamos el grado
            self.X = self.map_features(grado)                                               # Aumentamos features
            self.n, self.m = self.X.shape                                                   # Actualizamos n: features (filas), m: ejemplos (columnas)

        self.feature_norm = feature_norm                                                    # Variable de control que indica si se ha hecho feature normalization
        
        if feature_norm:
            self.aplicar_feature_normalization()                                            # Aplicar feature normalization

        if split is not None and test is None:                                              # Si se ha introducido algun valor en split
            self._train_test_split(split)                                                   # Separamos los datos
        
            if val_split is not None and val is None:                                       # Si se ha introducido algun valor en val_split
                self._train_val_split(val_split)                                            # Separamos en train y validation set
        
        if test is not None:                                                                # Si se ha pasado un test set como argumento
            self.X_test, self.y_test = test                                                 # Lo guardamos
            if axis == 1:                                                                   # En caso de que no sea nxm: transpuesta
                self.X_test = self.X_test.T
                self.y_test = self.y_test.T
        
        if val is not None:                                                                 # Si se ha pasado un test set como argumento
            self.X_val, self.y_val = val                                                    # Lo guardamos
            if axis == 1:                                                                   # En caso de que no sea nxm: transpuesta
                self.X_val = self.X_val.T
                self.y_val = self.y_val.T
                
        self.n, self.m = self.X.shape                                                       # Actualizamos la dimension de training set
        
        self.reg = reg                                                                      # Guardamos si se ha hecho regularizacion
        if self.reg and reg_par is not None:
            self.reg_par = reg_par                                                          # Guardamos el parametro de regularizacion

        self.theta = np.matrix(np.ones(self.n))                                            # Theta: peso de cada variable, sera una matriz 1xn


    """ 
    
        Funcion que crea features polinomiales a partir de unos features de entrada.
        
        -- grado: grado del polinomio a crear.
        -- X: entrada opcional de features.
    
    """
    
    def map_features(self, grado, X=None):
        
        if X is None:                               # Si no se introduce nada
            X = self.X                              # Se utilizan los features del modelo
        
        n, m = X.shape                              # Obtenemos las dimensiones
        
        array_aux = np.ones((grado + 1, m))         # Creamos una matriz de unos de la dimension: (g + 1) (features) x m (ejemplos)
        for i in range(1, grado + 1, 1):            # No recorremos el primer elemento
            array_aux[i, :] = np.power(X[1, :], i)  # Actualizamos el valor de los features: cada fila primer feature elevado a algo
            
        return array_aux
        
    
    """ 
    
        Funcion que calcula la normalizacion para cada uno de los elementos de la matriz de entrada.
        
        -- media: media de cada fila: todos los ejemplos de cada feature
        -- std: desviacio tipica de cada fila: todos los ejemplos de cada feature
        -- sub_media: la matriz de entrada menos la media
        -- normalizado: sub_media dividido por la desviacion típica
        -- X: entrada de features
    
    """

    def _feature_normalization(self, X):
        media = np.mean(X, axis=1)                                                          # Calculamos la media de cada fila, axis = 1
        n = media.shape[0]                                                                  # Obtenemos el numero de features
        media = np.reshape(media, (n, 1))                                                   # Obligamos a que sea una matriz nx1
        std = np.std(X, axis=1)                                                             # Calculamos la desviacion tipica de cada columna
        std = np.reshape(std, (n, 1))                                                       # Obligamos a que sea una matriz nx1
        sub_media = X - media                                                               # Restamos la media de cada columna a cada elemento de dicha columna
        normalizado = sub_media / std                                                       # Dividimos cada elemento de una columna por su desviacion tipica
        self.media, self.std = media, std                                                   # Guardamos la media y la desviacion tipica
        return normalizado                                                                  # Devolvemos el set normalizado


    """ 
    
        Funcion que aplica la normalizacion al modelo.
            
    """

    def aplicar_feature_normalization(self):
        X_norm = self._feature_normalization(self.X[1:])                        # Matriz normalizada que no incluye la fila de 1 correspondientes con x0
        self.X[1:] = X_norm                                                     # Guardamos el resultado
        #self.y = self._feature_normalization(self.y)                            # Normalizamos los targets
       

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
    
        Funcion que separa los datos en train set y validation set, se utilizara el primero para entrenar el modelo y el segundo
        para evaluar el rendimiento del mismo.
        
        -- num_train: int que indica el número de ejemplos que se incluiran en el train set.       
         
    """

    def _train_val_split(self, porcentaje):
        num_train = int((1 - porcentaje) * self.m)                                          # Calculamos cuantas columnas seran del training set
        self.X_val = self.X[:, num_train:]                                                  # Obtenemos las columnas hasta el limite
        self.y_val = self.y[:, num_train:]                                                  # Obtenemos las columnas hasta el limite
        self.X = self.X[:, :num_train]                                                      # Obtenemos las columnas restantes
        self.y = self.y[:, :num_train]                                                      # Obtenemos las columnas restantes
        self.n, self.m = self.X.shape                                                       # n: features (filas), m: ejemplos (columnas): actualizamos estos valores al cambiar X e y.

    
    """ 
    
        Funcion que calcula el coste en funcion del array de biases (peso de cada feature).
        
        -- h: evaluacion de la matriz de datos X
        -- error: diferencia entre la hipotesis y los valores reales
        -- error_sqr: elevar cada elemento al cuadrado
        -- J: suma todos los elementos la matriz hipotesis: matriz fila 1xm
        -- theta: entrada opcional de theta
        -- X: entrada opcional de features
        -- y: entrada opcional de targets
        -- reg_par: entrada opcional del parametro de regularizacion
    
    """

    def calcular_coste(self, theta=None, X=None, y=None, reg_par=None):
        
        if X is None:                                                                       # Si no se pasa nada por entrada
            X = self.X                                                                      # Utilizamos el guardado
        if y is None:                                                                       # Si no se pasa nada por entrada
            y = self.y                                                                      # Utilizamos el guardado
        if theta is None:                                                                   # Si no se pasa nada por entrada
            theta = self.theta                                                              # Utilizamos el guardado
        if reg_par is None and self.reg:                                                    # Si no se pasa nada por entrada y se ha utilizado regularizacion en el modelo
            reg_par = self.reg_par
            
        n, m = X.shape                                                                      # Obtenemos las dimensiones de los datos
            
        theta = theta.reshape((1, n));                                                      # Hacemos que theta sea un vector fila: 1 x n
        h = theta.dot(X)                                                                    # Evaluamos la funcion con todos los datos de entrada
        error = h - y                                                                       # Obtenemos el error
        error_sqr = np.power(error, 2)                                                      # Elevamos el error al cuadrado
        J = np.sum(error_sqr) / (2 * m)                                                     # Calculamos el coste
        if self.reg or reg_par is not None:                                                 # Si se ha indicado regularizacion
            regularizacion = (reg_par/(2*m))*np.sum(np.power(theta[0, 1:], 2))              # Aplicamos la formula de regularizacion
            J += regularizacion
        return J                                                                            # Devolvemos el resultado
    
    
    """ 
    
        Funcion que calcula el gradiente de cada uno de los biases en una matriz fila: 1xn.
        
        -- h: evaluacion de la matriz de datos X
        -- error: diferencia entre la hipotesis y los valores reales
        -- gradiente: multiplicar cada fila de error (todos los ejemplos), con cada columna de la transpuesta de la matriz de datos X
        -- theta: entrada opcional de theta
        -- X: entrada opcional de features
        -- y: entrada opcional de targets
        -- reg_par: entrada opcional del parametro de regularizacion
    
    """

    def _gradiente(self, theta=None, X=None, y=None, reg_par=None):
        
        if X is None:                                                                       # Si no se pasa nada por entrada
            X = self.X                                                                      # Utilizamos el guardado
        if y is None:                                                                       # Si no se pasa nada por entrada
            y = self.y                                                                      # Utilizamos el guardado
        if theta is None:                                                                   # Si no se pasa nada por entrada
            theta = self.theta                                                              # Utilizamos el guardado
        if reg_par is None and self.reg:                                                    # Si no se pasa nada por entrada y se ha utilizado regularizacion en el modelo
            reg_par = self.reg_par
            
        n, m = X.shape                                                                      # Obtenemos las dimensiones de los datos
        
        theta = theta.reshape((1, n));                                                      # Hacemos que theta sea un vector fila: 1 x n
        h = theta.dot(X)                                                                    # Evaluamos la funcion con todos los datos de entrada
        error = h - y                                                                       # Calculamos el error
        gradiente = (error.dot(X.T)) / m                                                    # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        if self.reg or reg_par is not None:
            gradiente[0, 1:] += (reg_par/m) * theta[0, 1:]                                  # Aplicar regularizacion
        return np.ravel(gradiente)   
    
    
    """ 
    
        Funcion que minimiza el coste utilizando la optimizacion de scipy.
        
        -- initial_theta: array de biases inicializado a cero
        -- Result: resultado de la minimizacion
        -- X: entrada opcional de features
        -- y: entrada opcional de targets
        -- reg_par: entrada opcional del parametro de regularizacion
    
    """

    def minimizacion(self, X=None, y=None, reg_par=None):
        
        if X is None:                                                   # Si no se ha introducido ningun valor
            X = self.X                                                  # Utilizamos el nuestro
        if y is None:                                                   # Si no se ha introducido ningun valor
            y = self.y                                                  # Utilizamos el nuestro
        
        n, m = X.shape                                                  # Obtenemos la dimension
        
        initial_theta = np.zeros(n);                                    # Inicializamos un array de variables auxiliares a cero

        Result = op.minimize(fun=self.calcular_coste,                   # Optimizamos la funcion de coste
                             x0=initial_theta,
                             args=(X, y, reg_par),
                             method='TNC',
                             jac=self._gradiente);

        self.theta = Result.x;                                          # Actualizamos el array de variables
    
    
    """ 
    
        Funcion que calcula la prediccion dada unos datos de entrada.
    
    """

    def prediccion(self, test):
        return np.ravel(self.theta).dot(test)
    
    
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
    
        Funcion que crea la grafica de los datos con la representación del modelo para polynomial
        regression.
    
    """   
    
    def plot_poly_regression(self, titulo, xlabel, ylabel):
        plt.scatter(x=np.ravel(self.X[1, :]), y=np.ravel(self.y[0, :]), color='red')        # Scatter plot de los datos
        plt.title(titulo)                                                                   # Titulo de la grafica
        plt.xlabel(xlabel)                                                                  # Etiqueta de las x
        plt.ylabel(ylabel)    

        val_min = np.min(self.X)                                                            # Obtenemos el menor valor de los datos
        val_max = np.max(self.X)                                                            # Obtenemos el mayor valor de los datos
        
        lista_ejemplo = np.matrix(np.linspace(val_min, val_max, num=50, dtype=float))       # Creamos una lista de 50 valores entre min y max
        n, m = lista_ejemplo.shape                                                          # Obtenemos las dimensiones
        lista_ejemplo = np.concatenate((np.matrix(np.ones(m)), lista_ejemplo))              # Ponemos unos
        lista_ejemplo = self.map_features(self.grado, lista_ejemplo)                        # Hacemos que sea polinomial
        
        plt.plot(np.ravel(lista_ejemplo[1, :]), np.ravel(self.prediccion(lista_ejemplo)))   # Grafica de preddicion: linear
        plt.show()

    
    """ 
    
        Funcion que obtiene la learning curve de nuestro modelo.
    
    """   
        
    def learning_curve(self):
        
        lista_costes_train = []                                                                     # Lista para guardar error de training set
        lista_costes_val = []                                                                       # Lista para guardar error de validation set
        
            
        for m in range(1, self.m + 1, 1):
            data = np.vstack((self.X, self.y))                                                      # Juntamos X e y
            random_data = data[:, np.random.choice(len(np.ravel(data[0])), size=m, replace=False)]  # Obtenemos indices aleatorios
            #X = random_data[:-1]                                                                    # Cogemos una porcion del training set
            #y = random_data[-1]                                                                     # Cogemos una porcion del training set
            X = self.X[:, :m]                                                                       # Cogemos una porcion del training set
            y = self.y[:, :m]                                                                       # Cogemos una porcion del training set
            self.minimizacion(X=X, y=y)                                                             # Aplicamos minimizacion
            lista_costes_train.append(self.calcular_coste(X=X, y=y, reg_par=0))                     # Guardamos el error de training set
            lista_costes_val.append(self.calcular_coste(X=self.X_val, y=self.y_val, reg_par=0))     # Guardamos el error de validation set
        
        lista = [m for m in range(1, self.m + 1, 1)]                                                # Lista de valores del 2 al 12
        
        plt.plot(lista, lista_costes_val, label="Validacion")                                       # Plot de valores de error de validation set
        plt.plot(lista, lista_costes_train, label="Train")                                          # Plot de valores de error de training set
        plt.title("Leaning curve con reg = %s" % self.reg_par)
        plt.xlabel("Numero de ejemplos")                                                            # Leyenda de las X
        plt.ylabel("Error")                                                                         # Leyenda de la y
        plt.legend()                                                                                # Ver leyendas
        plt.show()
        
            
    """ 
    
        Funcion itera sobre una lista de parametros de regularizacion para escoger aquel que sea obtenga
        menor coste en el validation set.
    
    """    
    
    def mejor_reg_par(self, lista_reg):
        lista_costes_train = []                                                                 # Inicializamos la lista de costes del training set
        lista_costes_val = []                                                                   # Inicializamos la lista de costes del validation set
        coste = 100                                                                             # Inicialimos el coste a un valor alto
        mejor_reg = None                                                                        # Inicializamos el valor del mejor parametro de regularizacion
        
        for valor in lista_reg:                                                                 # Cogemos una porcion del training set
            self.minimizacion(reg_par=valor)                                                    # Aplicamos minimizacion
            coste_train = self.calcular_coste(reg_par=0)                                        # Calculamos el coste sobre training set
            coste_validacion = self.calcular_coste(X=self.X_val, y=self.y_val,reg_par=0)        # Calculamos el coste sobre validation set
            if coste_validacion < coste:                                                        # Si es menor lo almacenamos
                coste = coste_validacion
                mejor_reg = valor
            lista_costes_train.append(coste_train)                                              # Guardamos el error de training set
            lista_costes_val.append(coste_validacion)                                           # Guardamos el error de validation set
        
        print("Mejor parámetro de regularizacion:", mejor_reg)                                  # Mostramos por pantalla
        
        self.minimizacion(reg_par=mejor_reg)   
        coste_test = self.calcular_coste(X=self.X_test, y=self.y_test, reg_par=0)               # Obtenemos el error sobre el test set
        
        print("Coste test para el mejor parámetro de regularizacion:", coste_test)              # Mostramos por pantalla
        
        plt.plot(lista_reg, lista_costes_val, label="Validacion")                               # Plot de valores de error de validation set
        plt.plot(lista_reg, lista_costes_train, label="Train")                                  # Plot de valores de error de training set
        plt.title("Mejor parametro de regularizacion")
        plt.xlabel("Parametro de regularizacion")
        plt.ylabel("Error")
        plt.legend()                                                                            # Ver leyendas
        plt.show()