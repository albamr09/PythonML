import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import math

GAUSSIAN = 'Gaussian'
MV_GAUSSIAN = 'Multivariate-Gaussian'


"""

------------------------------------------------------------------------------------------------------------------------

            Clase que aplica los algoritmos propios de AnomalyDetection para obtener anomalias dentro de un set
            de datos. Para ello se utilizará un set de training sobre el que se calculará la media y la varianza. 
            En base a estos datos calculamos un threshold (limite) sobre el cual dictar cuando se ha encontrado
            una anomalia.
            
            -- X: matriz de datos que se utilizara para entrenar.
            -- m: numero de ejemplos en el data set.
            -- n: numero de features en cada ejemplo.
            -- metodo: metodo para calcular la probabilidad dado un punto.
                -- Gaussian
                -- Multivariate Gaussian
            -- threshold: limite mínimo que indica si el dato es una anomalia.
            -- media: media de cada feature calculada a través del set de training.
            -- varianza: varianza de cada feature. (Metodo: Gaussian)
            -- cov_matrix: matriz de covarianza de todas las features. (Metodo: Multivariate Gaussian)

------------------------------------------------------------------------------------------------------------------------

"""



class AnomalyDetection:
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion de iniciacion de la clase, en la cual se inicializan las variables propias de la clase.

            -- axis: indicar como se estructuran los datos dentro de la matriz.
                -- 0: ejemplos en filas, features en columnas
            -- X: matriz de datos sobre la que realizar el entrenamiento.
            -- metodo: indica que metodo utilizar al calcular la probabilidad.
            -- threshold: limite minimo que indica si un dato es una anomalia.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def __init__(self, X, axis=0, metodo=GAUSSIAN, threshold=0.2):
        
        if axis == 0:
            self.X = X                                              # Almacenar matriz de datos de forma normal
            
        else:
            self.X = X.T                                            # Almacenar matriz de datos transpuesta
            
        self.m, self.n = self.X.shape                               # Almacenar dimension de datos: m (ejemplos), n (features)

        self.metodo = metodo                                        # Indicar el metodo para calcular la probabilidad
        self.threshold = threshold                                  # Indicar el limite minimo
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula los parametros necesarios para calcular la probabilidad de que un dato sea una anomalia.
            
            -- Gaussian:
                -- media de cada feauture (1xn)
                -- varianza de cada feature (1xn)
            -- Multivariate Gaussian:
                -- media de cada feature (1xn)
                -- matriz de covarianza (nxn)

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def fit(self):
        
        self.media = np.reshape(np.mean(self.X, axis=0), (-1, self.n))                          # Calculamos la media y aseguramos que tenga las dimensiones correctas
        
        if self.metodo == GAUSSIAN:
            self.varianza = np.reshape(np.var(self.X, axis=0), (-1, self.n))                    # Calculamos la varianza y aseguramos que tenga las dimensiones correctas

        elif self.metodo == MV_GAUSSIAN:
            self.cov_matrix = np.dot((self.X - self.media).T, (self.X - self.media))/self.m     # Calculamos la matriz de covarianza manualmente
            self.cov_matrix = np.cov(self.X.T, ddof=0)                                          # Calculamos la matriz de covarianza con numpy
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la probabilidad de un punto de ser una anomalia utilizando el metodo Gaussian.
            
            -- X: matriz de datos de entrada.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _calculo_prob_gaussian(self, X):  
        
        m, n = X.shape                                                              # Obtenemos las dimensiones
        exponente = -np.divide(np.power((X - self.media), 2), 2*self.varianza)      # Calculamos la parte de la formula que corresponde al exponente
        fit = np.multiply((1/np.sqrt(2*np.pi*self.varianza)), np.exp(exponente))    # Calculamos la probabilidad separada 
        p = np.reshape(np.prod(fit, axis=1), (m, -1))                               # Multiplicamos todas las probabilidades y aseguramos que tenga la dimension correcta
        return p  
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la probabilidad de un punto de ser una anomalia utilizando el metodo
            Multivariate Gaussian.
            
            -- X: matriz de datos de entrada.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _calculo_prog_multi_gaussian(self, X):
        
        m, n = X.shape                                                                                              # Obtenemos las dimensiones
        exponente = (-1/2)*np.diag(((X - self.media).dot(np.linalg.inv(self.cov_matrix).dot((X - self.media).T))))  # Calculamos la parte de la formula que corresponde al exponente
        fit = (1/((2*math.pi)**(n/2)*(np.linalg.det(self.cov_matrix))**(1/2)))                                      # Calculamos la primera parte de la formula
        p = np.reshape(fit * np.exp(exponente), (m, -1))                                                            # Aplicamos la formula completa y nos aseguramos que sea la dimension correcta
        return p


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que redirecciona a las funciones correspondientes dependiendo del metodo para calcula la
            probabilidad de un punto de ser una anomalia.
            
            -- X: matriz de datos de entrada.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def calculo_prob(self, X=None):
        
        if X is None:                                   # Si no se pasa X como argumento
            X = self.X                                  # Utilizar el almacenado
            
        if self.metodo == GAUSSIAN:                     # Calcular segun el metodo Gaussian
            return self._calculo_prob_gaussian(X)
        elif self.metodo == MV_GAUSSIAN:                # Calcular segun el metodo Multivariate Gaussian
            return self._calculo_prog_multi_gaussian(X)
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula la precision del modelo.
            
            -- true_positives: numero de ejemplos que se han etiquetado como positivo (anomalia) y son positivo.
            -- false_positives: numero de ejemplos que se an etiquetado como positivo (anomalia) y no son positivo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _calculo_precision(self, true_positives, false_positives):
        return (true_positives)/(true_positives + false_positives)          # Aplicar la formula de la precision


    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el recall del modelo.
            
            -- true_positives: numero de ejemplos que se han etiquetado como positivo (anomalia) y son positivo.
            -- false_negatives: numero de ejemplos que se an etiquetado como negativo (no anomalia) y no son negativo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _calculo_recall(self, true_positives, false_negatives):
        return (true_positives) / (true_positives + false_negatives)    # Aplicar la formula del recall

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que calcula el f1-score del modelo.
            
            -- true_positives: numero de ejemplos que se han etiquetado como positivo (anomalia) y son positivo.
            -- false_negatives: numero de ejemplos que se an etiquetado como negativo (no anomalia) y no son negativo.
            -- false_negatives: numero de ejemplos que se an etiquetado como negativo (no anomalia) y no son negativo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _f1_score(self, true_positives, false_positives, false_negatives):
        precision = self._calculo_precision(true_positives, false_positives)    # Calcular la precision
        recall = self._calculo_recall(true_positives, false_negatives)          # Calcular el recall
        return (2*precision*recall)/(precision + recall)                        # Aplicar la formula de f1-score
    
    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que obtiene el f1 score del modelo.
            
            -- epsilon: limite inferior que indica que ejemplos son anomalias.
            -- y_val: valores reales de los ejemplos, si es una anomalia o no.
            -- p: pobabilidades calculadas con el modelo.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def _calculo_f1_score(self, epsilon, y_val, p):
    
        prediccion = (p < epsilon)*1                    # Obtenemos nuestra prediccion a través de nuestras probabilidades y el limite inferior
        
        tp = np.sum((prediccion == 1) & (y_val == 1))   # Calculamos los true positives
        fp = np.sum((prediccion == 1) & (y_val == 0))   # Calculamos los false positives
        fn = np.sum((prediccion == 0) & (y_val == 1))   # Calculamos los false negatives

        return self._f1_score(tp, fp, fn)               # Devolvemos f1-score

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que obtiene el theshold que ofrece una mejor f1-score.
            
            -- X_val: matriz de datos.
            -- y_val: matriz de resultados reales sobre la que calcular la f1-score.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def minimizar_threshold(self, X_val, y_val):
        
        p = self.calculo_prob(X_val)                                # Calculamos las probabilidades sobre la matriz de datos
        best_F1 = 0                                                 # Inicializamos la f1-score
        step_size = (max(p) - min(p)) / 1000                        # Calculamos la distancia entre distintos thresholds a probar
        
        for epsilon in np.arange(min(p), max(p), step_size)[1:]:    # Probamos thresholds de la minima probabilidad a la maximo aumentado el valor de step en cada iteracion
            
            f1_score = self._calculo_f1_score(epsilon, y_val, p)    # Calculamos la f1-score
            
            if f1_score > best_F1:                                  # Si es mayor 
                best_F1 = f1_score                                  # Actualizamos la mejor f1-score
                self.threshold = epsilon                            # Actualizamos el limite
                
        return best_F1

    
    """

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que devulve las anomalias encontradas en un set de datos.
            
            -- X: matriz de datos sobre el que encontrar anomalias.

    ------------------------------------------------------------------------------------------------------------------------

    """
    
    def encontrar_anomalias(self, X):
        p = self.calculo_prob(X)                                                # Calcular probabilidades
        prediccion = (p < self.threshold)*1                                     # Obtener prediccion a través de probabilidades y threshold
        return X[np.where(prediccion == 1)[0]], np.where(prediccion == 1)[0]    # Devolver aquellos datos cuya prediccion es positiva, y lista de indices en los que la prediccion ha sido positiva