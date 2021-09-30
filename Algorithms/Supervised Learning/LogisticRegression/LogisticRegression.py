import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pandas as pd


"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos de Logistic Regression, es decir, algoritmos de clasificacion.
            
            -- X: matriz de terminos independientes.
            -- y: matriz fila de termino dependiente.
            -- n: numero de features, numero de filas.
            -- m: numero de ejemplos, numero de columnas.
            -- reg: boolean indica si se aplica regularizacion.
            -- theta: matriz fila de biases.

------------------------------------------------------------------------------------------------------------------------

"""


class LogisticRegression():

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: si 0 -> features en filas y ejemplos en columnas, si no viceversa.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def __init__(self, X, y, reg=False, axis=0, reg_par=None):
        if axis == 0:
            self.X = X                                                                                          # Inicializamos X
            self.y = y                                                                                          # Inicializamos y
        else:                                                                                                   # Si X e y no está en el formato correcto
            self.X = X.T                                                                                        # Inicializamos X
            self.y = y.T                                                                                        # Inicializamos y

        self.n, self.m = self.X.shape                                                                           # Guardamos las dimensiones

        if reg:                                                                                                 # Si se indica aplicar regularizacion
            self.X = self._map_feature(self.X)
            self.reg_par = reg_par                                                                              # Guardar el parametro de parametrizacion
        else:
            self.X = np.concatenate((np.matrix(np.ones(self.m)), self.X))                                       # Si no se quiere aplicar regularización se añaden 1

        self.reg = reg                                                                                          # Se guarda si se quiere regularizar
        self.n, self.m = self.X.shape                                                                           # Obtenemos la nueva dimension de la matriz de datos
        self.theta = np.matrix(np.zeros(self.n))                                                                # Inicializamos los biases


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que crea un vector de 28 elementos a partir de un vector de 2 elementos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _map_feature(self, X):
        n, m = X.shape                                                                                          # Obtenemos la dimension de la matriz de datos
        if (n == 2):                                                                                            # Si tiene dos features
            degree = 6;                                                                                         # El grado del polinomio
            mapeado = np.ones(m)                                                                                # Creamos una fila de 1, termino independiente
            for i in range(1, degree + 1):
                for j in range(0, i + 1):
                    multiplicacion = np.ravel(np.power(X[0, :], (i - j))) * np.ravel(np.power(X[1, :], (j)))    # Calculo de polinomio
                    mapeado = np.vstack((mapeado, multiplicacion))                                              # Lo añadimos al resultado
            return mapeado                                                                                      # Devolvemos la matriz mapeada
        else:
            print("Solo es un mapeado valido para dos features")                                                # Mensaje de error si hay más o menos que dos features


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que evalua segun la funcion sigmoid los valores independientes de la matriz X.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _sigmoid(self):
        self.theta = self.theta.reshape((1, self.n));                                                           # Hacemos que theta sea un vector fila: 1 x n
        z = self.theta.dot(self.X)                                                                              # Calculamos la entrada a la funcion sigmoid: theta*X
        z = 1 / (1 + np.exp(-z))                                                                                # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))
        return z


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que evalua segun la funcion sigmoid los valores independientes de la matriz X, para la funcion de 
            minimización, que requiere la introducción de argumentos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _sigmoid_min(self, X, theta):
        theta = theta.reshape((1, self.n));                                                           # Hacemos que theta sea un vector fila: 1 x n
        z = theta.dot(X)                                                                              # Calculamos la entrada a la funcion sigmoid: theta*X
        z = 1 / (1 + np.exp(-z))                                                                      # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))
        return z


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste tras evaluar la matriz de elementos independientes y el vector de biases.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def calculo_coste(self):
        z = self._sigmoid().T                                                               # Calculamos la hipotesis: en este caso la funcion sigmoid: transpuesta para permitir la multiplicacion con y
        sum = self.y.dot(np.log(z)) + (1 - self.y).dot(np.log(1 - z))                       # Aplicacions la funcion de coste simplificada
        if self.reg:                                                                        # En caso de haber aplicado regularizacion
            sum_reg = self.reg_par * np.sum(np.power(self.theta[0, 1:], 2)) / (2 * self.m)  # Calculo de la regularizacion para evitar overfitting
            return -np.ravel(sum)[0] / self.m + sum_reg                                     # Devolvemos el sumatorio entre el número de muestras
        else:
            return -np.ravel(sum)[0] / self.m                                               # Si no se aplica regularizacion no se añade


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste tras evaluar la matriz de elementos independientes y el vector de biases para la 
            funcion de minimizacion, ya que este requiere de argumentos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _calculo_coste_min(self, theta, X, y, reg_par=None):
        theta = theta.reshape((1, self.n));                                         # Hacemos que theta sea un vector fila: 1 x n
        z = self._sigmoid_min(X, theta).T                                           # Calculamos la hipotesis: en este caso la funcion sigmoid: transpuesta para permitir la multiplicacion con y
        sum = y * np.log(z) + (1 - y) * np.log(1 - z)                               # Aplicacions la funcion de coste simplificada
        if self.reg:                                                                # En caso de haber aplicado regularizacion
            sum_reg = reg_par * np.sum(np.power(theta[0, 1:], 2)) / (2 * self.m)    # Calculo de la regularizacion para evitar overfitting
            return -np.ravel(sum)[0] / self.m + sum_reg                             # Devolvemos el sumatorio entre el número de muestras
        else:
            return -np.ravel(sum)[0] / self.m                                       # Si no se aplica regularizacion no se añade


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente para aplicar el descenso.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _gradiente(self):
        h = self._sigmoid()                                                         # Aplicamos la funcion sigmoid
        error = h - self.y                                                          # Calculamos el error
        gradiente = (error * self.X.T) / self.m                                     # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        if self.reg:                                                                # En caso de que se haya aplicado regularizacion
            regularizacion = (self.reg_par / self.m) * self.theta                   # Calculamos la regularizacion de todas las variables independientes
            gradiente[0, 1:] = gradiente[0, 1:] + regularizacion[0, 1:]             # Sumamos gradiente y regularizacion excepto theta0, al que no se le aplica regularizacion
        return np.ravel(gradiente)                                                  # Hacemos que gradiente sea un vector fila: 1 x n


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente para aplicar el descenso para la minimizacion, que requiere de argumentos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _gradiente_min(self, theta, X, y, reg_par=None):
        h = self._sigmoid_min(X, theta)                                             # Aplicamos la funcion sigmoid
        error = h - y                                                               # Calculamos el error
        gradiente = (error * X.T) / self.m                                          # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        if self.reg:                                                                # En caso de que se haya aplicado regularizacion
            regularizacion = (reg_par / self.m) * theta                             # Calculamos la regularizacion de todas las variables independientes
            gradiente = gradiente.reshape((1, self.n));                             # Hacemos que gradiente sea un vector fila: 1 x n
            regularizacion = regularizacion.reshape((1, self.n));                   # Hacemos que regularizacion sea un vector fila: 1 x n
            gradiente[0, 1:] = gradiente[0, 1:] + regularizacion[0, 1:]             # Sumamos gradiente y regularizacion excepto theta0, al que no se le aplica regularizacion
        return np.ravel(gradiente)                                                  # Hacemos que gradiente sea un vector fila: 1 x n


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que aplica el algoritmo de descenso de gradiente para calcular el vector de biases más optimo.
            
            -- lr: learning rate.
            -- iter: numero de iteraciones.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def gradient_descent(self, lr, iter):
        coste_anterior = self.calculo_coste()                               # Creamos la variable que contendra el coste de la ronda anterior
        for i in range(iter):
            theta = self.theta - lr * self._gradiente()                     # Calculamos la nueva theta
            coste_actual = self.calculo_coste()                             # Calculamos el nuevo coste
            if (coste_actual > coste_anterior):                             # Si el nuevo coste es mayor, paramos
                break
            else:
                self.theta = theta                                          # Actualizamos theta
                coste_anterior = coste_actual                               # Actualizamos el coste de la ronda anterior como preparacion para la siguiente ronda


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion crea una gráfica a partir de los datos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def plot_datos(self, titulo, xlabel, ylabel, markers, color_label):

        fig, ax = plt.subplots()

        for marker in np.unique(markers):                                                                       # Para cada categoria hacemos un scatter
            ax.scatter(np.ravel(self.X[1, :])[markers == marker], np.ravel(self.X[2, :])[markers == marker],
                        marker=marker,
                        color=color_label[marker]['color'], label=color_label[marker]['label'])

        plt.title(titulo)                                                                                       # Titulo de la grafica
        plt.xlabel(xlabel)                                                                                      # Leyenda de las x
        plt.ylabel(ylabel)                                                                                      # Leyenda de las y
        plt.legend()                                                                                            # Establecemos la leyenda
        plt.show()                                                                                              # Visualizamos la grafica


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que crea una grafica de los datos y de la linea: decision boundary.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def plot_resultados(self, titulo, xlabel, ylabel, markers, color_label):

        self.theta = self.theta.reshape((1, self.n));                                                           # Hacemos que theta sea un vector fila: 1 x n
        fig, ax = plt.subplots()

        for marker in np.unique(markers):
            ax.scatter(np.ravel(self.X[1, :])[markers == marker], np.ravel(self.X[2, :])[markers == marker],    # Para cada categoría hacemos un scatter
                       marker=marker,
                       color=color_label[marker]['color'], label=color_label[marker]['label'])

        plt.title(titulo)                                                                                       # Titulo de la grafica
        plt.xlabel(xlabel)                                                                                      # Leyenda de las x
        plt.ylabel(ylabel)                                                                                      # Leyenda de las y

        if self.reg:                                                                                            # Si se aplica regularizacion
            u = np.linspace(-1, 1.5, 50)                                                                        # Creamos un vector de 50 elementos
            v = np.linspace(-1, 1.5, 50)                                                                        # Creamos un vector de 50 elementos
            z = np.zeros((len(u), len(v)))                                                                      # Inicializamos una matriz de 50 elementos a 0

            for i in range(len(u)):
                for j in range(len(v)):
                    tmp = np.array([u[i:i + 1], v[j:j + 1]])
                    tmp = self._map_feature(tmp)                                                                # Aplicamos la regularizacion a una matriz de ejemplo
                    z[i, j] = np.ravel(self.theta.dot(tmp))[0]                                                  # Evaluamos la matriz

            z = z.T                                                                                             # Transpuesta
            u, v = np.meshgrid(u, v)
            cs = ax.contour(u, v, z, levels=[0])                                                                # Contour de los datos calculados
            cs.collections[0].set_label("Decision boundary")                                                    # Establecemos la leyenda de decision boundary
        else:                                                                                                   # Si no se aplica regularizacion
            X_plot = np.array([np.min(self.X[1, :]) - 2, np.max(self.X[1, :]) + 2])                             # Un vector del elemento minimo y maximo
            y_plot = (-1 / self.theta[0, 2]) * (self.theta[0, 1] * X_plot + self.theta[0, 0])                   # Evaluamos el vector
            plt.plot(X_plot, y_plot, label="Decision boundary")                                                 # Creamos la linea de decision boundary

        plt.legend()
        plt.show()                                                                                              # Visualizamos la grafica


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que minimiza el coste, calculande el vector de biases más optimo.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def minimize(self):
        initial_theta = np.zeros(self.n);                               # Inicializamos una theta inicial a cero

        if not self.reg:                                                # Si no se ha aplicado regularizacion
            Result = op.minimize(fun=self._calculo_coste_min,           # Funcion a minimizar
                                 x0=initial_theta,                      # Primer argumento
                                 args=(self.X, self.y),                 # Demas argumentos
                                 method='TNC',
                                 jac=self._gradiente_min);
        else:                                                           # Si se aplica regularizacion
            Result = op.minimize(fun=self._calculo_coste_min,
                                 x0=initial_theta,
                                 args=(self.X, self.y, self.reg_par),   # Incluir el parametro de regularizacion como argumento
                                 method='TNC',
                                 jac=self._gradiente_min);

        self.theta = Result.x;                                          # Actualizamos theta


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que minimiza el coste, calculande el vector de biases más optimo utilizando la ecuacion normal.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def norm_ecuacion(self):
        X = self.X.T                                                                        # Es necesario que las features esten en columnas en lugar de en filas
        y = self.y.T

        if self.reg:                                                                        # Si se ha aplicado regularizacion
            m_reg = np.identity(self.n)                                                     # Creamos la matriz identidad
            m_reg[0, 0] = 0                                                                 # El primer elemento de la matriz es cero

            self.theta = np.linalg.inv(X.T.dot(X) + self.reg_par*m_reg).dot(X.T).dot(y)     # Resolvemos la ecuacion
            self.theta = self.theta.reshape((1, self.n))                                    # Hacemos que theta sea un vector fila: 1 x n
        else:
            self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)                          # Resolvemos la ecuacion de la normal
            self.theta = self.theta.reshape((1, self.n))                                    # Hacemos que theta sea un vector fila: 1 x n
    

"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos de Logistic Regression, es decir, algoritmos de clasificacion permitiendo
            varias categorias.
            
            -- X: matriz de terminos independientes.
            -- y: matriz de terminos dependiente.
            -- y_pred: matriz original.
            -- n: numero de features, numero de filas.
            -- m: numero de ejemplos, numero de columnas.
            -- c: numero de categorias.
            -- reg: boolean indica si se aplica regularizacion.
            -- reg_par: parametro de regularizacion.
            -- theta: matriz fila de biases.

------------------------------------------------------------------------------------------------------------------------

"""


class MultiLogisticRegression():
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: si 0 -> features en filas y ejemplos en columnas, si no viceversa.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def __init__(self, X, y, reg=False, axis=0, reg_par=None, categorias=None):
        if axis == 0:
            self.X = X                                                                                          # Inicializamos X
            self.y_prec = y                                                                                     # Guardamos los targets originales
        else:                                                                                                   # Si X e y no está en el formato correcto
            self.X = X.T                                                                                        # Inicializamos X
            self.y_prec = y.T                                                                                   # Guardamos los targets originales
        
        self.n, self.m = self.X.shape                                                                           # Guardamos las dimensiones
        
        self.y = np.reshape(y, (1, self.m))
        self.y = np.array(pd.get_dummies(np.ravel(y))).T
        self.c, self.m = self.y.shape
                                                                                       
        if reg:                                                                                                 # Si se indica aplicar regularizacion
            self.reg_par = reg_par                                                                              # Guardar el parametro de parametrizacion

        self.X = np.concatenate((np.matrix(np.ones(self.m)), self.X))                                           # Si no se quiere aplicar regularización se añaden 1
        self.categorias = categorias                                                                            # Guardar los nombres de las categorias
        self.reg = reg                                                                                          # Se guarda si se quiere regularizar
        self.n, self.m = self.X.shape                                                                           # Obtenemos la nueva dimension de la matriz de datos
        self.theta = np.matrix(np.zeros((self.c, self.n)))                                                      # Inicializamos los biases


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que evalua segun la funcion sigmoid los valores independientes de la matriz X.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _sigmoid(self):
        z = self.theta.dot(self.X)                                                                              # Calculamos la entrada a la funcion sigmoid: theta*X
        z = 1 / (1 + np.exp(-z))                                                                                # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))
        return z


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que evalua segun la funcion sigmoid los valores independientes de la matriz X, para la funcion de 
            minimización, que requiere la introducción de argumentos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _sigmoid_min(self, theta, X):
        z = theta.dot(X)                                                                                        # Calculamos la entrada a la funcion sigmoid: theta*X
        z = 1 / (1 + np.exp(-z))                                                                                # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))
        return z

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste tras evaluar la matriz de elementos independientes y el vector de biases.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def calculo_coste(self, indice_coste=None):
        z = self._sigmoid().T                                                               # Calculamos la hipotesis: en este caso la funcion sigmoid: transpuesta para permitir la multiplicacion con y
        sum = self.y.dot(np.log(z)) + (1 - self.y).dot(np.log(1 - z))                       # Aplicacions la funcion de coste simplificada
        sum = np.diagonal(sum)
        if self.reg:                                                                        # En caso de haber aplicado regularizacion
            sum_reg = self.reg_par * np.sum(np.power(self.theta[0, 1:], 2)) / (2 * self.m)  # Calculo de la regularizacion para evitar overfitting
            sum = -sum / self.m + sum_reg                                                   # Devolvemos el sumatorio entre el número de muestras
        else:
            sum = -sum/self.m
        sum = np.reshape(sum, (1, self.c))                                                  # Obligamos que sum sea un vector con un elemento por cada categoria
        if not indice_coste:                                                                # Si no se indica alguna categoria en concreto
            return sum                                                                      # Devolver el vector
        else:
            return sum[0, indice_coste]                                                     # Devolver un elemento


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste tras evaluar la matriz de elementos independientes y el vector de biases para la 
            funcion de minimizacion, ya que este requiere de argumentos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _calculo_coste_min(self, theta, X, y, reg_par=None):
        theta = np.reshape(theta, (self.c, self.n))
        z = self._sigmoid_min(theta, X).T                                     # Calculamos la hipotesis: en este caso la funcion sigmoid: transpuesta para permitir la multiplicacion con y
        sum = y.dot(np.log(z)) + (1 - y).dot(np.log(1 - z))                   # Aplicacions la funcion de coste simplificada
        sum = np.diagonal(sum)                                                          # Nos quedamos solo con la diagonal de la matriz
        if self.reg:                                                                    # En caso de haber aplicado regularizacion
            sum_reg = reg_par * np.sum(np.power(theta[0, 1:], 2)) / (2 * self.m)   # Calculo de la regularizacion para evitar overfitting
            sum = -sum / self.m + sum_reg                                               # Devolvemos el sumatorio entre el número de muestras
        else:
            sum = -sum/self.m
        sum = np.reshape(sum, (1, self.c))
        return np.sum(sum)
    

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente para aplicar el descenso.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _gradiente(self):
        h = self._sigmoid()                                                         # Aplicamos la funcion sigmoid
        error = h - self.y                                                          # Calculamos el error
        gradiente = (error * self.X.T) / self.m                                     # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        if self.reg:                                                                # En caso de que se haya aplicado regularizacion
            regularizacion = (self.reg_par / self.m) * self.theta                   # Calculamos la regularizacion de todas las variables independientes
            gradiente[:, 1:] = gradiente[:, 1:] + regularizacion[:, 1:]             # Sumamos gradiente y regularizacion excepto theta0, al que no se le aplica regularizacion
        return gradiente                    


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente para aplicar el descenso para la minimizacion, que requiere de argumentos.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _gradiente_min(self, theta, X, y, reg_par=None):
        theta = np.reshape(theta, (self.c, self.n))
        h = self._sigmoid_min(theta, X)                                         # Aplicamos la funcion sigmoid
        error = h - y                                                           # Calculamos el error
        gradiente = (error * X.T) / self.m                                      # Cada columna es el gradiente de una theta, variable distinta ya que se multiplica por X.T para sumatorio
        if self.reg:                                                            # En caso de que se haya aplicado regularizacion
            regularizacion = (self.reg_par / self.m) * theta                    # Calculamos la regularizacion de todas las variables independientes
            gradiente[:, 1:] = gradiente[:, 1:] + regularizacion[:, 1:]         # Sumamos gradiente y regularizacion excepto theta0, al que no se le aplica regularizacion
        return gradiente                                                        


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que aplica el algoritmo de descenso de gradiente para calcular el vector de biases más optimo.
            
            -- lr: learning rate.
            -- iter: numero de iteraciones.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def gradient_descent(self, lr, iter):
        for i in range(iter):
            self.theta = self.theta - lr * self._gradiente()                     # Calculamos la nueva theta
    

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que minimiza el coste, calculande el vector de biases más optimo.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def minimize(self):
        initial_theta = np.matrix(np.zeros((self.c, self.n)));          # Inicializamos una theta inicial a cero

        if not self.reg:                                                # Si no se ha aplicado regularizacion
            Result = op.minimize(fun=self._calculo_coste_min,
                                x0=initial_theta,
                                args=(self.X, self.y),                  # Incluir el parametro de regularizacion como argumento
                                method='TNC',
                                jac=self._gradiente_min);
        else:                                                           # Si se aplica regularizacion
            Result = op.minimize(fun=self._calculo_coste_min,
                                 x0=initial_theta,
                                 args=(self.X, self.y, self.reg_par),   # Incluir el parametro de regularizacion como argumento
                                 method='TNC',
                                 jac=self._gradiente_min);

        self.theta = Result.x;                                          # Actualizamos theta
        self.theta = np.reshape(self.theta, (self.c, self.n))
    

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que minimiza el coste, calculande el vector de biases más optimo utilizando la ecuacion normal.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def norm_ecuacion(self):
        X = self.X.T                                                                            # Es necesario que las features esten en columnas en lugar de en filas
        y = self.y.T
        det = np.linalg.det(X.T.dot(X))                                                         # Calculamos el determinante de la matriz a invertir
        if det > 0:
            if self.reg:                                                                        # Si se ha aplicado regularizacion
                m_reg = np.identity(self.n)                                                     # Creamos la matriz identidad
                m_reg[0, 0] = 0                                                                 # El primer elemento de la matriz es cero
                self.theta = np.linalg.inv(X.T.dot(X) + self.reg_par*m_reg).dot(X.T).dot(y)     # Resolvemos la ecuacion
            else:
                self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)                          # Resolvemos la ecuacion de la normal
        else:
            print("Matriz no inversible")


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que obtiene la categoria a la que pertenece un determinado conjunto de datos: ejemplo.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def prediccion(self, X_test):
        
        predicciones = self._sigmoid_min(self.theta, X_test)                    # Obtenemos la probabilidad de pertenecer a cada categoria
        indice = np.argmax(predicciones)                                        # Obtenemos la mayor probabilidad
        if self.categorias:                                                     # Si se han definido probabilidades
            return self.categorias[indice]                                      # Devolvemos la categoria correspondiente
        else:
            return indice                                                       # Devolvemos el indice


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la precision de nuestro modelo.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def precision(self):
        predicciones = self._sigmoid_min(self.theta, self.X)                    # Obtenemos las predicciones hechas por todos los modelos
        indices = np.argmax(predicciones, axis=0).T + 1                         # Obtenemos la prediccion mas alta y le sumamos 1
        igual = np.sum(indices == self.y_prec)                                  # Comprobamos cuantas coinciden con el original
        return igual / self.m                                                   # Devolvemos la precision: correctas / total
