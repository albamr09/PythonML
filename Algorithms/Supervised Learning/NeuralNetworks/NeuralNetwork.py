import numpy as np
import pandas as pd
import scipy.optimize as op

"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos de Logistic Regression, es decir, algoritmos de clasificacion pero
            desde una perspectiva de red neuronal, creando varios nodos y varias capas. Cada nodo funcionará como
            un clasificador independiente.
            
            -- X: matriz de terminos independientes.
            -- y: matriz fila de termino dependiente.
            -- y_hot_enc: codificacion de cada categorias en vectores de ceros, con un uno en la categoria a la que corresponde.
            -- n: numero de features, numero de filas.
            -- m: numero de ejemplos, numero de columnas.
            -- c: numero de categorias.
            -- theta: matriz fila de biases.
            -- epsilon_init: valor para inicializacion de theta, la aletoriedad de la inicializacion rompe la simetria.
            -- epsilon_gradient_check: valor que los vale para realizar un calculo de prueba.
            -- numero_capas: numero de capas en la red neuronal.

------------------------------------------------------------------------------------------------------------------------

"""

class NeuralNetwork():

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: si 0 -> features en filas y ejemplos en columnas, si no viceversa.
            -- epsilon_init: se indica un valor por defecto.
            -- epsilon_gradient_check: se indica un valor por defecto.
            -- reg: boolean que indica si se va a utilizar regularizacion en este modelo
            -- reg_par: parametro de regularizacion
            -- random: ordenar y de forma aleatoria

    ------------------------------------------------------------------------------------------------------------------------

    """

    def __init__(self, X, y, axis=0, epsilon_init=0.12, epsilon_gradient_check=10**(-4), reg=False, reg_par=None, split=None, random=False):

        if axis == 0:
            self.X = X                                                                          # Inicializamos normal si la entrada tiene el formato correcto: features en filas y ejemplos en columnas
        else:                                               
            self.X = X.T                                                                        # Si el formato es al reves almacenamos la transpuesta

        self.n, self.m = self.X.shape                                                           # Obtenemos las dimensiones de la entrada
        self.X = np.concatenate((np.matrix(np.ones(self.m)), self.X))                           # Añadimos una fila de unos, bias independiente
        self.n, self.m = self.X.shape                                                           # Actualizamos las dimensiones de la entrada
        self.y = np.reshape(y, (-1, self.m))                                                    # Forzamos a que y sea una matriz del tipo: 1 x m
        self._categorical()                                                                     # Codificamos las categorias
        if random:
            data = np.concatenate((self.X, self.y), axis=0)                                     # Agrupamos los datos
            np.take(data, np.random.permutation(data.shape[1]), axis=1, out=data)               # Ordenamos y de forma aleatoria
            self.y = data[-1, :]                                                                # Las predicciones estan en la última fila
            self.X = data[:-1, :]                                                               # Los datos estan en el resto
        self.c = np.unique(y).size                                                              # Obtenemos el numero de categorias
        self.y_hot_enc = np.array(pd.get_dummies(np.ravel(self.y))).T                           # Creamos una matriz en la cual cada columna es un ejemplo, y cada fila indica si pertenece (1) a la categoria o no (0)
        self.theta = []                                                                         # Creamos una lista vacia de pesos
        self.epsilon_init = epsilon_init                                                        # Inicializamos la variable de inicializacion aleatoria
        self.epsilon_gradient_check = epsilon_gradient_check                                    # Inicializamos la variable de chequeo
        self.numero_capas = 0                                                                   # Inicializamos el numero de capas
        self.reg = reg                                                                          # Inicializamos la variable de control de la regularizacion
        if self.reg and reg_par is not None:                                                    # Si se indica regularizacion
            self.reg_par = reg_par                                                              # Inicializamos el parametro de regularizacion
        if split is not None:                                                                   # Si se ha indicado un porcentaje de separacion
            self.test_split = True                                                              # Se ha hecho split en test y train
            self._train_test_split(porcentaje=split)                                            # Actualizar los datos
        else:
            self.test_split = False                                                             # Se ha hecho split en test y train
    

    """ 
    
    ------------------------------------------------------------------------------------------------------------------------

            Funcion que codifica las categorias de un set a numeros. 

    ------------------------------------------------------------------------------------------------------------------------     
             
    """

    def _categorical(self):
        categorias = pd.Categorical(self.y[0], categories=np.unique(self.y)).codes              # Codificacion de y en base a las distintas categorias
        categorias = np.array(categorias, dtype=int)                                            # Cast a numpy array
        self.y = np.reshape(categorias, (-1, self.m))                                           # Obligar a que sea un vector 1 x m

    
    """ 
    
    ------------------------------------------------------------------------------------------------------------------------

            Funcion que separa los datos en train set y test set, se utilizara el primero para entrenar el modelo y el segundo
            para evaluar el rendimiento del mismo.
        
            -- porcentaje: indica la cantidad de informacion que se utilizara para test.  

    ------------------------------------------------------------------------------------------------------------------------     
         
    """

    def _train_test_split(self, porcentaje):
        num_train = int((1 - porcentaje) * self.m)                                          # Calculamos cuantas columnas seran del training set
        self.X_test = self.X[:, num_train:]                                                 # Obtenemos las columnas hasta el limite
        self.y_test = self.y[:, num_train:]                                                 # Obtenemos las columnas hasta el limite
        self.X = self.X[:, :num_train]                                                      # Obtenemos las columnas restantes
        self.y = self.y[:, :num_train]                                                      # Obtenemos las columnas restantes
        self.n, self.m = self.X.shape                                                       # n: features (filas), m: ejemplos (columnas): actualizamos estos valores al cambiar X e y.
        self.y_hot_enc = self.y_hot_enc[:, :num_train]                                      # Actualizamos la variable que ha codificado las categorias

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que genera una matriz de valores con la dimension indicada en el argumento. Los valores del array
            oscilaran entre [-epsilon_init, epsilon_init]

            -- shape: dimension que tendra el array: n x m
                -- n: numero de nodos.
                -- m: features (nodos) de la capa anterior.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def random_init_theta(self, shape):
        theta = np.random.random_sample((shape)) * 2 * self.epsilon_init - self.epsilon_init    # Generamos numeros aleatorios y los acotamos con la variable epsilon_init
        return theta

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que inicializa las varibles necesarias para crear una nueva capa en nuestra red neuronal.

            -- shape: dimension que queremos que tenga la capa: n x m
                -- n: numero de nodos.
                -- m: numero de features (nodos) de la  capa anterior.


    ------------------------------------------------------------------------------------------------------------------------

    """

    def anadir_capa(self, shape):
        theta = self.random_init_theta(shape)   # Inicializamos theta
        self.theta.append(theta)                # La añadimos a la lista de pesos
        self.numero_capas += 1                  # Actualizamos el numero de capas
    

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que crea una lista, dividiendo los elementos de cada capa.

            -- theta: entrada, vector de una dimension que pasara a ser una lista con el theta de cada capa.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _roll_theta(self, theta):
        theta_reshape = []                                              # Creamos la lista temporal
        offset = 0                                                      # Inicializamos el offset
        for theta_element in self.theta:
            c, n = theta_element.shape                                  # Obtenemos el número de nodos (c), y el número de features en cada nodo (n)
            theta_temp = np.reshape(theta[offset:offset + c*n], (c, n)) # Obtener la matriz con reshape de dimensiones cxn (el offset indica los elementos ya recorridos)
            offset += c*n                                               # Actualizamos el offset + numero de elementos ya recorridos
            theta_reshape.append(theta_temp)                            # Añadir a la lista
        return theta_reshape

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que crea un vector de todos los elementos de la lista input.

            -- theta: lista con theta de cada capa que pasará a ser un vector de una dimension.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _unroll_theta(self, theta):
        theta_ravel = []                                # Creamos la lista que almacenará las matrices tras flatten
        for theta_element in theta:
            theta_ravel.append(np.ravel(theta_element)) # Hacer que la matriz sea un vector y almacenarlo en lista temporal
        return np.concatenate(theta_ravel)              # Hacer que la lista temporal sea un solo vector
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la funcion sigmoid dada unos pesos y unas entradas.

            -- theta: matriz de pesos.
            -- X: matriz de entrada.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def sigmoid(self, theta, X):
        z = theta.dot(X)                                                                                        # Calculamos la entrada a la funcion sigmoid: theta*X
        a = 1 / (1 + np.exp(-z))                                                                                # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))
        return a

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente de sigmoid dados unos pesos y una entrada.

            -- theta: matriz de pesos.
            -- X: matriz de entrada.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def sigmoid_gradient(self, theta, h):
        sig_grad = np.multiply(h, (1 - h))                          # Aplicar la ecuacion para calcular el gradiente
        return sig_grad


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que se encargar de realizar todos los calculos entre los nodos y las capas de la red neuronal para
            obtener el resultado final de la categoria.

            -- theta: entrada opcional de theta.
            -- capa: entrada opcional para calcular la salida de una capa en concreto, si no se especifica la funcion 
                    retornará la salida de la red neuronal.
            -- test: variable de control que indica si utilizar o no X_test

    ------------------------------------------------------------------------------------------------------------------------

    """

    def feed_forward(self, theta=None, capa=None, test=False):

        if theta is None:                                                   # Si no se introduce theta como argumento
            theta = self.theta                                              # Inicializar theta con el almacenado en el objeto

        if test:                                                            # Si se indica utilizar X_test
            a = self.X_test
            n, m = self.X_test.shape                                        # Guardar dimensiones de test
        else:
            a = self.X                                                      # La primera entrada es X
            n, m = self.X.shape                                             # Guardar dimensiones de train

        if capa is not None:                                                # Si se ha indicado una capa
            if capa <= len(theta) and capa >= 0:                            # Chequeamos que la capa esta dentro de los limites
                for i in range(capa):                                       # Recorremos las capas
                    a = self.sigmoid(theta[i], a)                           # Calculamos la salida de la capa
                    a = np.concatenate((np.matrix(np.ones(m)), a))          # Añadimos una fila de unos
                return a
            else:
                print("El número de capa no es válido")                     # Mensaje de error
        else:
            for elemento in theta:
                a = self.sigmoid(elemento, a)                               # Calculamos la salida de la capa actual
                a = np.concatenate((np.matrix(np.ones(m)), a))              # Añadimos una fila de unos
            h = a[1:, :]                                                    # Eliminamos los 1 en la última capa
            return h


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste, es decir el grado de error con respecto a los datos pero aplicando regularizacion.

            -- theta: argumento opcional de theta, lista o vector de pesos.
            -- unrolled: indica si el theta introducido es una lista (False), o un vector de una dimension (True)

    ------------------------------------------------------------------------------------------------------------------------

    """

    def calculo_coste(self, theta=None, unrolled=False):

        if theta is None:                                                                                                   # Si no se introduce theta como argumento
            theta = self.theta                                                                                              # Inicializar theta con el almacenado en el objeto

        if unrolled:                                                                                                        # Si theta se ha flatten en un vector de una dimension
            theta = self._roll_theta(theta)                                                                                 # Crear lista con matriz theta de capa capa

        h = self.feed_forward(theta)                                                                                        # Obtener la salida para todos los ejemplo
        coste = -np.sum(np.diagonal(self.y_hot_enc.T.dot(np.log(h)) + (1 - self.y_hot_enc.T).dot(np.log(1 - h))))/self.m    # Calcular el error con la matriz codificada de y

        if self.reg:                                                                                                        # Si se ha indicado que se aplica regularizacion
            reg_parcial = 0                                                                                                 # Inicializamos la variable temporal
            for elemento in theta:                                                                                          # Para capa
                reg_parcial += np.sum(np.power(elemento[:, 1:], 2))                                                         # No sumar el término independiente en cada nodo: primera fila
            reg_result = self.reg_par/(2*self.m)*(reg_parcial)                                                              # Calcular la regularizacion
            coste = coste + reg_result
        
        return coste


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la precision del modelo.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def precision_modelo(self):
        if self.test_split:                                                                             # Si se ha hecho split sobre los datos utilizar test
            h = self.feed_forward(test=True)
            y = np.array(self.y_test, dtype=int)                                                        # Casteamos array a int                     
        else:
            h = self.feed_forward()
            y = np.array(self.y, dtype=int)                                                             # Casteamos array a int

        p = np.argmax(h, axis = 0)                                                                      # Obtenemos el indice del elemento de mayor valor
        return np.mean(p == y) * 100                                                                    # Calculamos cuantos valores coinciden con el valor a predecir                       


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que aplica el algoritmo de backpropagation para actualizar los pesos de cada capa de la red
            neuronal.

            -- theta: argumento opcional de una lista o vector de pesos.
            -- unrolled: indica si theta es una lista (False) o un vector de una dimension (True).
            -- unroll: indica si la salida será una lista (False) o un vector de una dimension (True).

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def back_propagation(self, theta=None, unrolled=False, unroll=False):

        if theta is None:                                                                               # Si no se ha indicado ningun theta como argumento
            theta = self.theta                                                                          # Inicializar theta con el almacenado en el objeto
        
        if unrolled:
            theta = self._roll_theta(theta)                                                             # Creamos una lista del array

        delta = []                                                                                      # Inicializamos las lista temporal que contendra el delta de cada nodo
        delta_sum = []                                                                                  # Inicializamos la lista temporal que contendra el sumatorio delta
        gradientes = []                                                                                 # Inicializamos la lista que contendrá los gradientes de cada capa

        h = self.feed_forward(theta=theta)                                                              # Calculamos el valor del la salida para empezar a propagar hacia atras   

        delta_next = h - self.y_hot_enc                                                                 # Calculamos el primer delta: el de la ultima capa
        delta.append(delta_next)                                                                        # Lo añadimos a la lista temporal
        indice = self.numero_capas - 1                                                                  # El indice indica hasta que capa calcular la salida
        
        for elemento in reversed(theta[1:]):                                                            # Recorremos las capas de atras hacia adelante
            h = self.feed_forward(theta=theta, capa=indice)                                             # Calculamos la salida de la capa actual
            delta_aux = np.multiply(elemento.T.dot(delta_next), self.sigmoid_gradient(elemento, h))     # Aplicamos la formula del gradiente
            delta_next = delta_aux[1:, :]                                                               # No cogemos el elemento independiente
            delta.append(delta_next)                                                                    # Lo añadimos a la lista de delta
            indice -= 1                                                                                 # Actualizamos el indice
        
        delta.reverse()                                                                                 # Damos la vuelta a la lista
        for indice in range(len(delta)):                                
            h = self.feed_forward(theta=theta, capa=indice)                                             # Obtenemos la salida de cada capa
            delta_sum.append(delta[indice].dot(h.T))                                                    # Añadimos (delta * a) a la lista de delta_mayuscula -> sumatorio

        for indice in range(len(delta_sum)):
            gradiente = (1/self.m) * delta_sum[indice]                                                  # Calculamos el grandiente: delta_mayuscula / m
            if self.reg:
                gradiente[1:, :] += (self.reg_par/self.m) * theta[indice][1:, :]                        # Si se indica regularizacion aplicarla: no regularizan primer elemento
            gradientes.append(gradiente)                                                                # Lo añadimos a la lista

        coste = self.calculo_coste(theta=theta)

        if unroll:                                                                                      # Si se ha indicado que se quiere hacer flatten a un vector de una dimension
            return coste, self._unroll_theta(gradientes)
        else:
            return coste, gradientes

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente aproximado.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def calculo_gradiente_aproximado(self):
        unrolled_theta = self._unroll_theta(self.theta)                                             # Crear un vector a partir de la lista
        numgrad = np.zeros(unrolled_theta.shape)                                                    # Crear un vector de ceros
        perturb = np.diag(self.epsilon_gradient_check * np.ones(unrolled_theta.shape))  
        for i in range(unrolled_theta.size):
            coste1 =  self.calculo_coste(theta=unrolled_theta - perturb[:, i], unrolled=True)       # Calculamos el coste de theta modificado
            coste2 =  self.calculo_coste(theta=unrolled_theta + perturb[:, i], unrolled=True)       # Calculamos el coste de theta modificado
            numgrad[i] = (coste2 - coste1)/(2*self.epsilon_gradient_check)                          # Aplicamos la formula y lo guardamos
        
        return self._roll_theta(numgrad)                                                            # Retornamos el gradiente aproximado

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que comprueba que el algoritmo de backpropagation esta bien implementado.

            -- check: boolean que indica si presentar los vector de gradiente de cada metodo por pantalla

    ------------------------------------------------------------------------------------------------------------------------

    """

    def gradient_checking(self, check=False):       
        n_entradas = 3                                                                                                      # Inicializamos las variables de la prueba
        n_features = 5
        n_categorias = 3
        n_nodo_intermedio = 3
        X = self.random_init_theta((n_features, n_entradas))
        y = np.reshape(np.arange(1, 1 + n_entradas) % n_categorias, (1, n_entradas))

        nn = NeuralNetwork(X, y)                                                                                            # Creamos la red neuronal
        nn.anadir_capa((n_nodo_intermedio, n_features + 1))                                                                 # Añadimos una capa
        nn.anadir_capa((n_categorias, n_nodo_intermedio + 1))                                                               # Añadimos una capa

        gradiente_aproximado = self._unroll_theta(nn.calculo_gradiente_aproximado())                                        # Calculamos el gradiente aproximado
        gradiente_real = self._unroll_theta(nn.back_propagation()[1])                                                       # Calculamos el gradiente real

        #Visualizacion

        if check:
            print(np.stack([gradiente_aproximado, gradiente_real], axis=1))

        
        diff = np.linalg.norm(gradiente_aproximado - gradiente_real)/np.linalg.norm(gradiente_aproximado + gradiente_real)  # Calcular diferencia
        print('Consideraremos la implementacion como correcta si el valor de la diferencia es menor a 1e-9.\n\n'
            'Diferencia relativa: %g\n' % diff)
    

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion minimiza el coste modificando la lista de pesos.

            -- iter: numero de iteraciones maximas para minimizar el coste.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def minimizacion(self, iter):

        initial_theta = []                                      # Creamos la lista de pesos

        for i in self.theta:
            c, n = i.shape
            theta_temp = self.random_init_theta((c, n))         # La inicializamos de forma aleatoria
            initial_theta.append(theta_temp)

        initial_theta = self._unroll_theta(initial_theta)       # Creamos un vector de una dimension a partir de la lista

        Result = op.minimize(fun=self.back_propagation,         # Metodo que se minimizara
                                 x0=initial_theta,              # Argumento que se modificara
                                 method='L-BFGS-B',             # Metodo de minimizacion
                                 args=(True, True),             # Indica que la entrada y la salida son vectores
                                 jac=True, 
                                 options={'maxiter': iter});    # Numero de iteraciones del algoritmo
        
        self.theta = self._roll_theta(Result.x);                # Actualizamos theta