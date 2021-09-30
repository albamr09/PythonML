import numpy as np
import time


"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos de KMeans, es decir, algoritmos de clasificacion para realizar 
            clustering.
            
            -- X: matriz de terminos independientes.
            -- K: numero de cluster en los que se intentará agrupar los datos.
            -- n: numero de features, numero de columnas.
            -- m: numero de ejemplos, numero de filas.
            -- centroids: puntos que actuarán como referencia para cada cluster
            -- indices: array que indica a que centroid pertenece cada ejemplo

------------------------------------------------------------------------------------------------------------------------

"""


class KMeans:
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: si 0 -> features en columnas y ejemplos en filas, si no viceversa.
            -- random: inicializar de forma aleatoria los centroids, asignándolo a un ejemplo.
            -- centroids: indica si se pasa una lista de centroids como argumento.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def __init__(self, X, K=None, axis = 0, random=True, centroids=None):
        if axis == 0:
            self.X = X                                      # Inicializamos X de forma normal
        else:
            self.X = X.T                                    # Almacenamos la transpuesta de X para que las filas sean ejemplo y las columnas features.
            
        self.m, self.n = self.X.shape                       # Guardamos el numero de ejemplos (m) y el número de de features (n)
        
        if K is None:                                       # Inicializar al k más optimo
            self._elbow_k()
        else:
            self.K = K                                      # Almacenamos el numero de clusters
        
        if centroids is not None:                           # Si se ha pasado una lista de centroids por argumento
            if (self.K, self.n) == centroids.shape:         # Si tiene las dimensiones adecuadas: K clusters cada uno con n features.
                self.centroids = centroids                  # Lo guardamos
        else:
            if random:                                      # Si se ha indicado que la inicializacion sea aleatoria
                self.centroids = self.init_centroids()
            else:
                self.centroids = np.zeros((self.K, self.n)) # Si no inicializar a cero
        
        self.indices = np.zeros(self.m, dtype=int)          # Inicializar los indices a cero: un indice por cada ejemplo (m)
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que obtiene el número de clusters mínimo que ofrece un coste menor en comparación
             con otras opciones.
             
             -- epsilon_init: diferencia minima admitida.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _elbow_k(self, epsilon_init=0.2):
        
        k_actual = 1
        J_actual = 0
        
        kmeans = KMeans(self.X, k_actual)
        kmeans.fit(evitar_local_optima=True)
        J_anterior = kmeans.J   
        k_actual += 1
        
        while True:
            kmeans = KMeans(self.X, k_actual)
            kmeans.fit(evitar_local_optima=True)
            J_actual = kmeans.J
            
            diferencia = J_anterior - J_actual
                
            if diferencia < epsilon_init:
                self.K = k_actual
                break
            else:
                k_actual += 1
                J_anterior = J_actual
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que inicializa de forma aleatoria los centroids, es decir iguala su valor a un ejemplo
            aleatorio.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def init_centroids(self):
        millis = int(round(time.time() * 1000))
        np.random.seed(seed=millis%(2**32 - 1))
        indices_random = np.random.permutation(self.m)          # Reordenamos de forma aleatoria los indices de X, de 0 a m
        return self.X[indices_random[:self.K]]                  # Devolvemos los K primeros indices
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que asigna el centroid más cercano a cada ejemplo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def asignar_centroid(self):
        
        J_total = 0                                                                 # Inicializamos el coste a cero
    
        for i in np.arange(self.m):                                                 # Por cada ejemplo
            J = np.sqrt(np.sum(np.square(self.X[i] - self.centroids), axis = 1))    # Calculamos la distancia a todos los centroid: J (1xk)
            self.indices[i] = np.argmin(J)                                          # Obtenemos el indice del elemento de menor valor
            J_total += np.power(J[self.indices[i]], 2)                              # Calculamos el coste y lo añadimos al total
        
        return (1/self.m) * J_total                                                 # Aplicamos el resto de la fórmula


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion actualiza la posicion de los centroids en funcion de los ejemplos asignados al mismo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def actualizar_centroids(self):

        for i in range(self.K):                                                     # Para cada centroid
            self.centroids[i] = np.mean(self.X[self.indices == i], axis = 0)        # Calculamos la media de los puntos asignados

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Aplicamos el algoritmo de K-means:
            
            1. Asignamos un centroid a cada ejemplo.
            2. Una vez asignado recalculamos el valor del centroid en funcion de los puntos que 
                se le hayan asignado.
            
            En cada iteracion del algoritmo guardamos los centroids calculados en la misma
            
            -- iter: numero de iteraciones por las que pasará el algoritmo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _algoritmo_kmeans(self, iter):
        
        lista_centroids = np.zeros((iter, self.K, self.n))          # Inicializamos la lista de centroids a 0
        
        for i in range(iter):
            self.J = self.asignar_centroid()                                 # Paso 1 del algoritmo
            self.actualizar_centroids()                             # Paso 2 del algoritmo
            lista_centroids[i] = self.centroids.copy()              # Añadimos los centroids a la lista
            
        return lista_centroids
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Aplicamos el algoritmo de K-means utilizando la repeticion para evitar errores por 
            minimos locales. Para ello creamos listas auxiliares que almacenan los resultados 
            de cada repeticion y seleccionamos el que minimo coste nos ofrezca.
            
            1. Asignamos un centroid a cada ejemplo.
            2. Una vez asignado recalculamos el valor del centroid en funcion de los puntos que 
                se le hayan asignado.
            
            En cada iteracion del algoritmo guardamos los centroids calculados en la misma
            
            -- iter: numero de iteraciones por las que pasará el algoritmo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _algoritmo_kmeans_rep(self, num_repeticiones, iter):
        
        lista_costes = np.zeros((1, num_repeticiones))                  # Iniliciación de listas auxiliares.
        lista_aux_centroids = []
        lista_aux_indices = []
        lista_lista_centroids = []
        
        for rep in range(num_repeticiones):
            
            self.centroids = self.init_centroids()                      # Inicializamos los centroids de nuevo aleatoriamente
            lista_centroids = np.zeros((iter, self.K, self.n))          # Inicializamos la lista de centroids a 0
            
            for i in range(iter):
                J = self.asignar_centroid()                             # Paso 1 del algoritmo
                self.actualizar_centroids()                             # Paso 2 del algoritmo
                lista_centroids[i] = self.centroids.copy()              # Añadimos los centroids a la lista

            lista_costes[0, rep] = J                                    # Almacenar coste
            lista_aux_centroids.append(self.centroids.copy())           # Alcacenar centroids
            lista_aux_indices.append(self.indices.copy())               # Almacenar indices
            lista_lista_centroids.append(lista_centroids.copy())        # Alacenar alteracion de los centroids segun la iteracion
        
        indice_mejor_coste = np.argmin(lista_costes)                    # Obtenemos el indice del resultado que haya ofrecido un mejor coste
        self.centroids = lista_aux_centroids[indice_mejor_coste]        # Obtenemos los valores correspondientes a dicho indice
        self.indices = lista_aux_indices[indice_mejor_coste]
        lista_centroids = lista_lista_centroids[indice_mejor_coste]
        self.J = lista_costes[0, indice_mejor_coste]
        
        return lista_centroids


    """

    ------------------------------------------------------------------------------------------------------------------------

            Aplica un distinto algoritmo segun si se quiera llevar a cabo varias veces para evitar errores
            relacionados con los mínimo locales.
            
            -- evitar_local_optima: variable de control que indica si se lleva a cabo la repeticion.
            -- num_repeticiones: indica el numero de veces que se repetirá el algoritmo con centroids
                                 escogidos aleatoriamente

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def fit(self, iter=10, evitar_local_optima=False, num_repeticiones=50):
        
        if evitar_local_optima:   
            return self._algoritmo_kmeans_rep(num_repeticiones, iter)           # Algoritmo que pretende evitar errores por minimos locales
         
        else:   
            return self._algoritmo_kmeans(iter)                                 # Algoritmo original
            
        