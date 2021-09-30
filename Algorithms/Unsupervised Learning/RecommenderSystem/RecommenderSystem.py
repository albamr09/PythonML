import numpy as np
import scipy.optimize as op


OPTIMO_COBA = 'CoBa'
OPTIMO_COFI = 'CoFi'


"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos de Linear Regression, es decir, algoritmos de predicción
            de valores. Este tipo de modelo almacenará los pesos de varios usuarios para predecir la valoración
            sobre un item basandose en valoraciones anteriores del mismo usuario y en la valoración de otros 
            usuarios sobre ese item.
            
            -- Y: matriz (nm x nu) que contiene las valoraciones de cada usuario sobre cada item.
            -- Y_original: copia de la matriz de valoraciones original.
            -- R: matriz (nm x nu) que tiene 1 si el usuario i ha valorado el item j y 0 en caso contrario.
            -- media: la puntuacion media de cada item (nm x 1).
            -- X: matriz de datos de cada ejemplo (nm x n).
            -- theta: matriz pesos de cada usuario (nu x n).
            -- nm: numero de items.
            -- nu: numero de usuarios.
            -- n: numero de features.
            -- epsilon_gradient_check: pequeño valor que nos permite aproximar el gradiente
            -- epsilon_init: valor minimo y maximo sobre el que se generan los valores aleatorios de X y theta.
            -- reg_par: parametro de regularizacion para prevenir overfitting.
            -- optim: indica el algoritmo de optimizacion a utilizar.
            -- lr: learning rate con el que se aplica el gradiente en el descenso gradiente.

------------------------------------------------------------------------------------------------------------------------

"""

class RecommenderSystem:
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: si 0 -> ejemplos en filas y usuarios en columnas si no, al contrario.
            -- reg_par: argumento opcional que inicializa el parametro de regularizacion.
            -- optim: argumento opcional que inicializa el metodo de optimizacion.
            -- X: argumento opcional que inicializa la matriz de datos.
            -- theta: argumento opcional que inicializa la matriz de pesos.
            -- epsilon_gradient_check: argumento opcional que inicializa la variable que permite calcular una 
                                        aproximacion del gradiente.
            -- epsilon_init: argumento opcional que limita a que valores se puede inicializar X y theta.
            -- n_features: en caso de no pasar por argumento X y theta 

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def __init__(self, Y, R, axis=0, reg_par=1, optim=OPTIMO_COBA, X=None, theta=None, epsilon_gradient_check=10**(-4), epsilon_init=0.12, n_features=10, lr=0.01):
        
        if axis == 0:
            self.Y = Y                                                                              # Si la matriz es nm x nu (ejemplos en filas y usuarios en columnas) se almacena normal
            self.R = R
        else:
            self.Y = Y.T                                                                            # Si no se almacena la transpuesta
            self.R = R.T

        self.Y_original = self.Y.copy()                                                             # Guardamos una copia de la matriz original de valoraciones
        self._mean_normalization()                                                                  # Realizamos mean normalization para evitar accuracy engañoso
            
        self.X = X                                                                                  # Almacenamos la matriz de datos
        self.theta = theta                                                                          # Almacenamos la matriz de pesos
        
        self.nm, self.nu = self.Y.shape                                                             # Almacenamos el tamaño
        
        self.epsilon_gradient_check = epsilon_gradient_check                                        # Almacenamos las variables de control
        self.epsilon_init = epsilon_init                                                            # Almacenamos las variables de control
          
        if theta is None and X is None:                                                             # Si ninguno se ha inicializado guardamos el numero de features pasado como argumento
            self.n = n_features
        
        if X is not None:                                                                           # Si X se ha pasado
            _, self.n = self.X.shape                                                                # Guardamos el numero de features
        if theta is not None:                                                                       # Si theta se ha pasado
            _, self.n = self.theta.shape                                                            # Guardamos el numero de features
        if X is None:                                                                               # Si X no se ha pasado
            self.X = self._random_init((self.nm, self.n))                                           # Lo inicilizamos de forma aleatoria
        if theta is None:                                                                           # Si theta no se ha pasado
            self.theta = self._random_init((self.nu, self.n))                                       # Lo inicializamos de forma aleatoria
        
        self.reg_par = reg_par                                                                      # Almacenamos variables de control
        self.optim = optim                                                                          # Almacenamos el metodo de optimizacion
        self.lr = lr                                                                                # Almacenamos variables de control
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que genera una matriz de valores con la dimension indicada en el argumento. Los valores del array
            oscilaran entre [-epsilon_init, epsilon_init]

            -- shape: dimension que tendra el array: n x m

    ------------------------------------------------------------------------------------------------------------------------

    """

    def _random_init(self, shape):
        
        vector = np.random.normal(size=(shape)) * 2 * self.epsilon_init - self.epsilon_init    # Generamos numeros aleatorios y los acotamos con la variable epsilon_init
        return vector
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la valoracion media de cada ejemplo sin tener en cuenta aquellos usuarios que no han 
            realizado valoracion: R[item, usuario] == 0.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _mean_normalization(self):
        self.media = np.reshape(np.divide(np.sum(self.Y_original, axis=1), np.sum(self.R, axis=1)), (-1, 1))    # Suma de todas las valoraciones/suma de todos los usuarios que han realizado una valoración
        self.Y = np.multiply((self.Y_original - self.media), self.R)                                            # Restarle la media y multiplicar por R: poner a cero los items que no han sido valorados.
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el error de la prediccion con respecto a las valoraciones finales.
            
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
    
    def _error(self, theta=None, X=None):
        
        if theta is None:                                                                               # Si no se pasa theta como argumento
            theta = self.theta                                                                          # Utilizamos el almacenado
        
        if X is None:                                                                                   # Si no se pasa X como argumento
            X = self.X                                                                                  # Utilizamos el almacenado
            
        return np.multiply(theta.dot(X.T).T,(self.R)) - np.multiply(self.Y, self.R)                     # Calculamos el error: prediccion*R - y*R
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula gradiente de cada theta.
            
            -- ravel: indica si hacer flatten sobre la matriz theta para obligar que sea un vector y concatenar el
                        gradiente de theta y el de X.
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
    
    def gradiente(self, ravel=False, theta=None, X=None):
        
        if X is None:                                                       # Si X no se indica
            X = self.X                                                      # Utilizar el almacenado
        
        if theta is None:                                                   # Si theta no se indica
            theta = self.theta                                              # Utilizar el almacenado
        
        
        if self.optim == OPTIMO_COBA:
            return self._gradiente_COBA(X=X, theta=theta)                   # Aplicar el algoritmo de Content Based
        elif self.optim == OPTIMO_COFI:
            return self._gradiente_COFI(ravel=ravel, X=X, theta=theta)      # Aplicar el algoritmo de Collaborative Filtering
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula gradiente de cada theta aplicando el algoritmo de Collaborative Filtering.
            
            -- ravel: indica si hacer flatten sobre la matriz theta para obligar que sea un vector y concatenar el
                        gradiente de theta y el de X.
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
    
    def _gradiente_COFI(self, ravel=False, theta=None, X=None):
        
        if X is None:                                                                   # Si X no se indica
            X = self.X                                                                  # Utilizar el almacenado
        
        if theta is None:                                                               # Si theta no se indica
            theta = self.theta                                                          # Utilizar el almacenado
        
        gradiente_theta = self._error(theta=theta, X=X).T.dot(X)                        # Calcular el gradiente de theta
        gradiente_theta += self.reg_par * theta                                         # Aplicar regularizacion
        
        gradiente_X = self._error(theta=theta, X=X).dot(theta)                          # Calcular el gradiente de X
        gradiente_X += self.reg_par * X                                                 # Aplicar regularizacion
        
        if ravel:
            return np.concatenate((np.ravel(gradiente_X), np.ravel(gradiente_theta)))   # Juntar ambos gradientes en un vector de una sola dimension
        else:
            return gradiente_theta, gradiente_X                                         # Devolver gradientes

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula gradiente de cada theta aplicando el algoritmo de Content Based.
            
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
    
    def _gradiente_COBA(self, theta=None, X=None):   
        
        if X is None:                                               # Si X no se indica
            X = self.X                                              # Utilizar el almacenado
        
        if theta is None:                                           # Si theta no se indica
            theta = self.theta                                      # Utilizar el almacenado
        
        gradiente = self._error(theta=theta, X=X).T.dot(X)          # Calcular el gradiente
        gradiente[:, 1:] += self.reg_par * theta[:, 1:]             # Aplicar regularizacion
        
        return gradiente
    
     
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste aplicando el algoritmo de Content Based.
            
            -- return_gradiente: indica si devolver el calculo de gradiente
            -- ravel: indica si hacer flatten sobre la matriz theta para obligar que sea un vector y concatenar el
                        gradiente de theta y el de X.
            -- params: pasar theta como vector de una dimension en lugar de una matriz nu x n.
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
     
    def _calculo_coste_COBA(self, return_gradiente=False, ravel=False, params=None, theta=None, X=None):
        
        if theta is None:                                                           # Si theta no se indica
            theta = self.theta                                                      # Utilizar el almacenado
            
        if X is None:                                                               # Si X no se indica
            X = self.X                                                              # Utilizar el almacenado
            
        if params is not None:                                                      # Si se indica algun valor el params
            theta = np.reshape(params, (self.nu, self.n))                           # Obligar que theta sea de la forma: nu x n
                
        coste = (1/2) * np.sum(np.power(self._error(theta=theta, X=X), 2))          # Calculamos el coste
        regularizacion = (self.reg_par/2) * np.sum(theta[:, 1:])                    # Calculamos la regularizacion
        coste = coste + regularizacion                                              # Aplicamos la regularizacion
        
        if return_gradiente:                                                        # Si se indica devolver el gradiente
            return coste, self.gradiente(ravel=ravel, X=X, theta=theta)             # Devolver: coste, gradiente
        else:
            return coste                                                            # Solo devolver coste
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el coste aplicando el algoritmo de Content Based.
            
            -- return_gradiente: indica si devolver el calculo de gradiente
            -- ravel: indica si hacer flatten sobre la matriz theta para obligar que sea un vector y concatenar el
                        gradiente de theta y el de X.
            -- params: pasar theta y X como vector de una dimension en lugar de una matriz theta(nu x n) y X(nm x n).
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
    
    def _calculo_coste_COFI(self, params=None, return_gradiente=False, ravel=False, theta=None, X=None):
        
        if theta is None:                                                                                       # Si theta no se indica
            theta = self.theta                                                                                  # Utilizar el almacenado
            
        if X is None:                                                                                           # Si X no se indica
            X = self.X                                                                                          # Utilizar el almacenado
            
        if params is not None:                                                                                  # Si se pasa params por argumento
            X = np.reshape(params[:self.nm*self.n], (self.nm, self.n))                                          # Obtener los primeros nm*n parametros del vector que corresponden a X y crear matriz nm x n
            theta = np.reshape(params[self.nm*self.n:], (self.nu, self.n))                                      # Obtener los siguientes nu*n parametros que corresponden a theta y crear matriz nu x n
        
        coste = (1/2) * np.sum(np.power(self._error(theta=theta, X=X), 2))                                      # Calcular coste
        regularizacion = (self.reg_par/2)*np.sum(np.power(X, 2)) + (self.reg_par/2)*np.sum(np.power(theta, 2))  # Calcular regularización
        coste = coste + regularizacion                                                                          # Aplicar regularizacion
        
        if return_gradiente:                                                                                    # Si se indica devolver gradiente
            return coste, self.gradiente(ravel=ravel, X=X, theta=theta)                                         # Devolver: coste, gradiente
        else:
            return coste                                                                                        # Solo devolver coste
          
          
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que gestiona el calculo del coste segun es Content Based o Collaborative Filtering.
            
            -- return_gradiente: indica si devolver el calculo de gradiente
            -- ravel: indica si hacer flatten sobre la matriz theta para obligar que sea un vector y concatenar el
                        gradiente de theta y el de X.
            -- params: pasar theta y X como vector de una dimension en lugar de una matriz theta(nu x n) y X(nm x n).
            -- theta: argumento opcional de matriz de pesos.
            -- X: argumento opcional de matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------
    
    """
    
    def calculo_coste(self, params=None, return_gradiente=False, ravel=False, theta=None, X=None):
        
        if theta is None:                                                                                                       # Si theta no se indica
            theta = self.theta                                                                                                  # Utilizar el almacenado
            
        if X is None:                                                                                                           # Si X no se indica
            X = self.X                                                                                                          # Utilizar el almacenado
        
        if self.optim == OPTIMO_COBA:
            return self._calculo_coste_COBA(params=params, return_gradiente=return_gradiente, ravel=ravel, theta=theta, X=X)    # Content Based
        elif self.optim == OPTIMO_COFI:
            return self._calculo_coste_COFI(params=params, return_gradiente=return_gradiente, ravel=ravel, theta=theta, X=X)    # Collaborative Filtering
       
       
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que aplica el algoritmo de descenso gradiente acorde al metodo Content Based.
            
            -- iter: numero de iteraciones a realizar.

    ------------------------------------------------------------------------------------------------------------------------
    
    """ 
        
    def _descenso_gradiente_COBA(self, iter=100):
        
        theta = self.theta.copy()                                           # Almacenamos una matriz de pesos auxiliar
        coste_anterior = self.calculo_coste(theta=theta)                    # Calculamos el coste actual

        for i in range(iter):
            theta = theta - self.lr * self.gradiente(theta=theta)           # Actualizamos theta segun el descenso gradiente
            coste_actual = self.calculo_coste(theta=theta)                  # Calculamos el coste

            if coste_anterior <= coste_actual:                              # Si el nuevo coste es mayor
                break                                                       # Parar
            else:
                self.theta = theta.copy()                                   # Actualizamos el theta original
                coste_anterior = coste_actual                               # El coste actual sera el coste anterior en la siguiente iteracion
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que aplica el algoritmo de descenso gradiente acorde al metodo Collaborative Filtering.
            
            -- iter: numero de iteraciones a realizar.

    ------------------------------------------------------------------------------------------------------------------------
    
    """ 
              
    def _descenso_gradiente_COFI(self, iter=100):
        
        theta = self.theta.copy()                                   # Almacenamos una matriz de pesos auxiliar
        X = self.X.copy()                                           # Almacenamos una matriz de datos auxiliar
        coste_anterior = self.calculo_coste(theta=theta, X=X)       # Calculamos el coste actual

        for i in range(iter):
            gradiente = self.gradiente(theta=theta, X=X)            # Obtenemos ambos gradientes
            theta = theta - self.lr * gradiente[0]                  # Actualizamos theta con su gradiente
            X = X - self.lr * gradiente[1]                          # Actualizamos X con su gradiente
            coste_actual = self.calculo_coste(theta=theta, X=X)     # Calculamos el nuevo coste

            if coste_anterior <= coste_actual:                      # Si es mayor que el de la ronda anterior
                break                                               # Parar
            else:
                self.theta = theta.copy()                           # Actualizar theta actual
                self.X = X.copy()                                   # Actualizar X actual
                coste_anterior = coste_actual                       # Actualizar coste de ronda anterior


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion maneja la aplicacion del descenso gradiente en funcion del método de optimizacion.
            
            -- iter: numero de iteraciones a realizar.

    ------------------------------------------------------------------------------------------------------------------------
    
    """ 
       
    def descenso_gradiente(self, iter=100):
        if self.optim == OPTIMO_COBA:               # Algoritmo de Content Based
            self._descenso_gradiente_COBA(iter)
        if self.optim == OPTIMO_COFI:               # Algoritmo de Collaborative Filtering
            self._descenso_gradiente_COFI(iter)

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el gradiente aproximado.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def calculo_gradiente_aproximado(self):

        numgrad_theta = np.zeros(self.theta.shape)                                                  # Inicializamos el gradiente aproximado de theta a cero                                                                                                                 
        
        for i in range(self.theta.shape[0]):                                                        # Recorremos cada usuario
            for j in range(self.theta.shape[1]):                                                    # Recorremos cada features

                theta_menos = self.theta.copy()                                                     # Creamos una copia de theta
                theta_mas = self.theta.copy()                                                       # Creamos una copia de theta
                theta_menos[i, j] -= self.epsilon_gradient_check                                    # Restamos un valor determinado al feature del usuaario
                theta_mas[i, j] += self.epsilon_gradient_check                                      # Sumamos un valor determinado al feature del usuaario
 
                coste_menos = self.calculo_coste(theta=theta_menos)                                 # Calculamos el coste de theta modificado
                coste_mas = self.calculo_coste(theta=theta_mas)                                     # Calculamos el coste de theta modificado
                numgrad_theta[i, j] = (coste_mas - coste_menos)/(2*self.epsilon_gradient_check)     # Aplicamos la formula de aproximacion y guardamos en la matriz de gradientes aproximados de theta
                
        if self.optim == OPTIMO_COFI:                                                               # Si se utiliza el metodo de optimizacion de Collaborative Filtering
            
            numgrad_X = np.zeros(self.X.shape)                                                      # Iniciliazamos el gradiente aproximado de X a cero
            
            for i in range(self.X.shape[0]):                                                        # Recorremos cada item
                for j in range(self.X.shape[1]):                                                    # Recorremos cada feature

                    X_menos = self.X.copy()                                                         # Creamos una copia
                    X_mas = self.X.copy()                                                           # Creamos una copia
                    X_menos[i, j] -= self.epsilon_gradient_check                                    # Restamos un valor determinado al feature del ejemplo
                    X_mas[i, j] += self.epsilon_gradient_check                                      # Sumamos un valor determinado al feature del ejemplo
    
                    coste_menos = self.calculo_coste(X=X_menos)                                     # Calculamos el coste con X modificado
                    coste_mas = self.calculo_coste(X=X_mas)                                         # Calculamos el coste con X modificado
                    numgrad_X[i, j] = (coste_mas - coste_menos)/(2*self.epsilon_gradient_check)     # Lo almacenamos en el gradiente aproximado de X
                    
            return numgrad_theta, numgrad_X                                                         # Devolvemos ambos gradientes
        
        return numgrad_theta                                                                        # Devolvemos el gradiente de theta
    

    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que comprueba que el gradiente esta bien calculado.

            -- check: boolean que indica si presentar los vector de gradiente de cada metodo por pantalla

    ------------------------------------------------------------------------------------------------------------------------

    """

    def gradient_checking(self, check=False): 
        
        n_usuarios = 3                                                                                                                      # Inicializamos las variables de la prueba
        n_features = 5
        n_peliculas = 3
        
        Y = self.Y[:n_peliculas, :n_usuarios]                                                                                               # Inicializamos matriz de valoraciones
        R = self.R[:n_peliculas, :n_usuarios]                                                                                               # Inicializamos matriz de valoraciones realizadas
        
        X = self.X[:n_peliculas, :n_features]                                                                                               # Inicializamos matriz de datos
        theta = self.theta[:n_usuarios, :n_features]                                                                                        # Inicializamos matriz de pesos

        rs = RecommenderSystem(Y, R, X=X, theta=theta, optim=self.optim, reg_par=1.5)

        gradiente_aproximado = rs.calculo_gradiente_aproximado()                                                                            # Calculamos el gradiente aproximado
        gradiente_real = rs.gradiente()                                                                                                     # Calculamos el gradiente real
        
        
        if self.optim == OPTIMO_COFI:                                                                                                       # Para el metodo de Collaborative Filtering comparamos los gradientes de theta y X
            
            if check:                                                                                                                       # Visualizar valores
                print("\nGradiente de theta\n")
                print(np.stack([gradiente_aproximado[0], gradiente_real[0]], axis=1))
                print("\nGradiente de X")
                print(np.stack([gradiente_aproximado[1], gradiente_real[1]], axis=1))
            
            print("\nGradiente de theta\n")
            diff = np.linalg.norm(gradiente_aproximado[0] - gradiente_real[0])/np.linalg.norm(gradiente_aproximado[0] + gradiente_real[0])  # Calcular diferencia
            print('Diferencia es menor a 1e-9.\n\n'
                'Diferencia relativa: %g\n' % diff)
            
            print("\nGradiente de X\n")
            diff = np.linalg.norm(gradiente_aproximado[1] - gradiente_real[1])/np.linalg.norm(gradiente_aproximado[1] + gradiente_real[1])  # Calcular diferencia
            print('Diferencia es menor a 1e-9.\n\n'
                'Diferencia relativa: %g\n' % diff)
            
        elif self.optim == OPTIMO_COBA:                                                                                                     # Para Content Based sólo trabajamos con el gradiente de theta
            
            if check:                                                                                                                       # Visualizar valores
                print("\nGradiente de theta\n")
                print(np.stack([gradiente_aproximado, gradiente_real], axis=1))
                
            print("\nGradiente de theta\n")
            diff = np.linalg.norm(gradiente_aproximado - gradiente_real)/np.linalg.norm(gradiente_aproximado + gradiente_real)              # Calcular diferencia
            print('Diferencia es menor a 1e-9.\n\n'
                'Diferencia relativa: %g\n' % diff)
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion minimiza el coste modificando la matriz de pesos theta y en caso de Collaborative Filtering la matriz
            de datos X.

            -- iter: numero de iteraciones maximas para minimizar el coste.
            -- metodo: metodo de minimizacion.

    ------------------------------------------------------------------------------------------------------------------------

    """

    def minimizacion(self, iter=None, metodo='TNC'):
   
        initial_theta = self._random_init((self.nu, self.n))                                    # Inicializar matriz de theta de forma aleatoria

        if self.optim == OPTIMO_COFI:                                                           # Para Collaborative Filtering
            
            if self.X is None:                                                                  # Si X no se ha inicializado
                initial_X = self._random_init((self.nm, self.n))                                # Inicializar con valores aleatorios
            else:
                initial_X = self.X                                                              # Utilizar la almacenada
                
            valores_iniciales = np.concatenate((np.ravel(initial_X), np.ravel(initial_theta)))  # Crear un vector con la concatenacion de X y theta
            
            Result = op.minimize(fun=self.calculo_coste,                                        # Metodo que se minimizara
                                    x0=valores_iniciales,                                       # Argumento que se modificara
                                    method=metodo,                                              # Metodo de minimizacion
                                    args=(True, True),                                          # Indica que la entrada y la salida son vectores
                                    jac=True, 
                                    options={'maxiter': iter});                                 # Numero de iteraciones del algoritmo
        
            self.X = np.reshape(Result.x[:self.nm*self.n], (self.nm, self.n))                   # Obtener los nm*n primeros elementos en forma de matriz nm x n
            self.theta = np.reshape(Result.x[self.nm*self.n:], (self.nu, self.n))               # Obtener los nu*n siguientes elementos en forma de matriz nu x n
        
        elif self.optim == OPTIMO_COBA:                                                         # Para Content Based
            
            valores_iniciales = np.ravel(initial_theta)                                         # Crear un vector a partir de la matriz de pesos
            
            Result = op.minimize(fun=self.calculo_coste,                                        # Metodo que se minimizara
                                    x0=valores_iniciales,                                       # Argumento que se modificara
                                    method=metodo,                                              # Metodo de minimizacion
                                    args=(True, False),                                         # Indica que la entrada y la salida son vectores
                                    jac=True,
                                    options={'maxiter': iter})                                  # Numero de iteraciones del algoritmo
            
            self.theta = np.reshape(Result.x, (self.nu, self.n))                                # Obtener la matriz de pesos
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula las predicciones de cada usuario sobre cada item a traves de la matriz de datos
            y la matriz de pesos.
            
            -- indice_usuario: indica el indice del usuario sobre el que se quiere realizar predicciones.
            -- excluir: indica si se realiza predicciones sobre todos los items o sólo sobre los valorados.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def predecir(self, indice_usuario=None, excluir=False):

        if indice_usuario is None:
            if not excluir:
                return self.X.dot(self.theta.T) + self.media                                                                # Predicciones sobre todos los usuarios y todos los items
            else:
                return np.multiply((self.X.dot(self.theta.T) + self.media), self.R)                                         # Predicciones sobre todos los usuarios solo de los items valorados
        else:
            if not excluir:
                return np.reshape(self.X.dot(self.theta.T)[:, indice_usuario], (-1, 1)) + self.media                        # Predicciones solo sobre todos los items para el usuario correspondiente a indice_usuario
            else:
                return np.multiply((np.reshape(self.X.dot(self.theta.T)[:, indice_usuario], (-1, 1)) + self.media),         # Predicciones solo sobre los items ya predichos para el usuario correspondiente a indice_usuario
                                   np.reshape(self.R[:, indice_usuario], (-1, 1))) 
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el modulo entre un item de la matriz de datos y el resto de datos de la matriz.
            
            -- item: vector sobre el que calcular los modulos.
            -- X: matriz de datos.

    ------------------------------------------------------------------------------------------------------------------------

    """
            
    def _distancia_item(self, item, X):
        modulo = np.reshape(np.sqrt(np.sum(np.power(item - X, 2), axis=1)), (1, -1))         # Calculamos el modulo de item con respecto a todos los items de X
        indices = np.argsort(modulo)                                                         # Obtenemos los indices ordenados de menor a mayor modulo
        return indices
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion obtiene el numero de items determinado mas similares segun la matriz de datos a un item indicado
            por el indice i.
            
            -- i: indice que indica sobre que item compara en la matriz de datos.
            -- numero: numero de items similares.
            -- verbose: imprimir el resultado.
            -- items_ids: identificador de los items (i.e.: nombre).

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def items_relacionados(self, i, numero=5, verbose=False, items_ids=None):
        
        item = self.X[i]                                                    # Obtenemos el item a comparar
        X = np.vstack((self.X[:i], self.X[(i + 1):]))                       # Lo excluimos de la matriz de datos
        indices = self._distancia_item(item=item, X=X)                      # Obtenemos los indices de los items mas similares
        
        if verbose and items_ids is not None:                               # Imprimir resultado
            print('\nTop 5 items relacionados con %s:' % items_ids[i])
            for indice in indices[0, :numero]:
                print('La pelicula %s' % (items_ids[indice]))
        

        return X[indices[0, :numero]], indices[0, :numero]                  # Devolvemos: datos correspondientes a items similares, indices de items similares
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion obtiene items a recomendar al usuario en funcion de valoraciones anteriores y valoraciones
            de otros usuarios.
            
            -- usuario: indice del usario al que realizar recomendaciones
            -- numero: numero de recomendaciones
            -- verbose: imprimir resultado
            -- items_ids: descripcion de los items (i.e.: nombre)

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def recomendaciones_usuario(self, usuario, numero=5, verbose=False, items_ids=None):
        
        if usuario < self.nu and usuario >= 0:
            
            prediccion = self.predecir(usuario).ravel()                                                                                             # Predicciones del usuario
            lista_usuario = np.multiply((np.reshape(self.Y[:, usuario], (-1, 1)) + self.media), np.reshape(self.R[:, usuario], (-1, 1))).ravel()    # Valoracion realizadas por el usuario
            
            if verbose and items_ids is not None:                                                                                                   # Mostrar resultado
                print('\nTop recomendaciones:')
                for indice in prediccion.argsort()[::-1][:numero]:                                                                                  # Ordenar lista de predicciones de mayor a menor (valoracion)
                    print('La valoracion es %.1f para la pelicula %s' % (prediccion[indice], items_ids[indice]))
                    
                print('Puntuaciones originales')              
                for indice in lista_usuario.argsort()[::-1][:numero]:                                                                               # Ordenar lista de valoraciones de mayor a menor
                    print('La valoracion es %.1f para la pelicula %s' % (lista_usuario[indice], items_ids[indice]))
                    
            return self.X[prediccion.argsort()[::-1][:numero]], prediccion.argsort()[::-1][:numero], prediccion                                     # Devolvemos: matriz de datos de items recomendados, indice de predicciones de items recomendados,  lista de predicciones de items recomendados 
        
        else:
            print('El usuario esta fuera de rango')
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que permite añadir un nuevo usuario al modelo.
            
            -- indice: posicion donde se añadira.
            -- valoraciones: lista de valoraciones opcional.
            -- entrenar: indica si entrenar el modelo tras añadir el usuario.
            -- iter: numero de iteraciones al entrenar.

    ------------------------------------------------------------------------------------------------------------------------

    """
            
    def anadir_usuario(self, indice, valoraciones=None, entrenar=True, iter=None):
        
        if valoraciones is None:                                                                                                        # Si no se pasa una lista de valoraciones
            valoraciones = np.zeros(self.nm)                                                                                            # Inicializar a cero
        
        self.Y_original = np.hstack((self.Y_original[:, :indice], valoraciones.reshape(-1, 1), self.Y_original[:, indice:]))            # Añadir valoraciones original del usuairo en el indice indicado
        self.R = np.hstack((self.R[:, :indice], (valoraciones != 0).reshape(-1, 1), self.R[:, indice:]))                                # Añadir si se ha hecho valoracion o no en el indice indicado

        nuevo_theta = self._random_init((1, self.n))                                                                                    # Inicializar theta de forma aleatoria
        self.theta = np.vstack((self.theta[:indice], nuevo_theta, self.theta[indice:]))                                                 # Introducir en la matriz de pesos

        self._actualizar_modelo(entrenar=entrenar, iter=iter)                                                                           # Actualizar modelo
        
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que permite añadir un nuevo item al modelo.
            
            -- indice: posicion donde se añadira.
            -- valoraciones: lista de valoraciones opcional.
            -- entrenar: indica si entrenar el modelo tras añadir el usuario.
            -- iter: numero de iteraciones al entrenar.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def anadir_item(self, indice, valoraciones=None, entrenar=True, iter=None):
        
        if valoraciones is None:                                                                                        # Si no se ha introducido
            valoraciones = np.zeros(self.nu)                                                                            # Inicializar a cero
        
        self.Y_original = np.vstack((self.Y_original[:indice], valoraciones.reshape(1, -1), self.Y_original[indice:]))  # Añadir a matriz de valoraciones orignal
        self.R = np.vstack((self.R[:indice], (valoraciones != 0).reshape(1, -1), self.R[indice:]))                      # Añadir a matriz de control de valoraciones

        nuevo_X = self._random_init((1, self.n))                                                                        # Inicializar X aleatoriamente
        self.X = np.vstack((self.X[:indice], nuevo_X, self.X[indice:]))                                                 # Introducir en la matriz
        
        self._actualizar_modelo(entrenar=entrenar, iter=iter)                                                           # Actualizar el modelo
        
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que permite actualizar el modelo tras añadir un usuario o item.
            
            -- entrenar: indica si entrenar el modelo tras añadir el usuario.
            -- iter: numero de iteraciones al entrenar.

    ------------------------------------------------------------------------------------------------------------------------

    """
        
    def _actualizar_modelo(self, entrenar=False, iter=None):
        
        self._actualizar_shapes()                   # Actualizar variables que almacenan las dimensiones del modelo
        self._mean_normalization()                  # Aplicar mean normalization
        
        if entrenar and iter is not None:           # Aplicar minimizacion
            self.minimizacion(iter=iter)

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que actualiza las dimensiones del modelo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _actualizar_shapes(self):
        
        self.nm, self.nu = self.Y_original.shape        # Guardar el numero de items y el numero de usuarios
        