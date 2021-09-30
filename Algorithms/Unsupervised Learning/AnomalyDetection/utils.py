import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que muestra una gráfica "Scatter" de los datos.

            -- X: matriz de dos dimensiones que representa la ubicación X e Y de los puntos.

    ------------------------------------------------------------------------------------------------------------------------

"""

def plotData(X):
    plt.scatter(X[:, 0], X[:, 1], marker='x')                           # Definimos la grafica
    plt.show()                                                          # La mostramos por pantalla
    

"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que muestra una gráfica "Scatter" de los datos resaltando las anomalias.

            -- X: matriz de dos dimensiones que representa la ubicación X e Y de los puntos.
            -- X_anom: matriz de dos dimensiones que contiene la informacion posicional de las
                        anomalias detectadas.

    ------------------------------------------------------------------------------------------------------------------------

"""

def plotDataAnom(X, X_anom):
    plt.scatter(X[:, 0], X[:, 1], label='Ejemplos normales')            # Definimos la gráfica de datos normales
    plt.scatter(X_anom[:, 0], X_anom[:, 1], label='Anomalias')          # Definimos la gráfica de las anomalías
    plt.legend()                                                        # Activar leyendas
    plt.show()                                                          # Mostrar por pantalla


"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que muestra varias graficas informativas:
            
                - Contour plot: grados de cercania con la media, a través de la varianza.
                - Contour plot 3D
                - Scatter plot de los datos

            -- X: matriz de dos dimensiones que representa la ubicación X e Y de los puntos.
            -- X_anom: matriz de dos dimensiones que contiene la informacion posicional de las
                        anomalias detectadas.

    ------------------------------------------------------------------------------------------------------------------------

"""    

def plotDistribucion(ad, X, X_anom=None):
    
    u = np.linspace(0, 30, 50)                                                                          # Creamos un vector de 50 elementos
    v = np.linspace(0, 30, 50)                                                                          # Creamos un vector de 50 elementos
    z = np.zeros((len(u), len(v)))                                                                      # Inicializamos una matriz de 50 elementos a 0

    for i in range(len(u)):
        for j in range(len(v)):
            tmp = np.array([u[i:i + 1], v[j:j + 1]]).T                                                  # Creamos un vector de datos de prueba
            p = ad.calculo_prob(tmp)                                                                    # Calculamos la probabilidad de ser anomalia
            z[i, j] = np.ravel(p)[0]                                                                    # Guardamos el resultado
    z = z.T
         
    fig = plt.figure()
    ax = plt.axes(projection='3d')                                                                      # Definimos el plot 3D
    ax.contour3D(u, v, z, 10.**np.arange(-21, -2, 3), cmap='rainbow')                                   # Contour plot de las probabilidades
    ax.contour3D(u, v, z, 50, cmap='rainbow')                                                           # Contour plot de las probabilidades
    ax.set_title('Gaussian Distribution')                                                               # Titulo
    plt.show()

    fig, ax = plt.subplots()    
    u, v = np.meshgrid(u, v)                                                                            # Creamos matriz
    ax.contourf(u, v, z,  50)                                                                           # Contour de los datos calculados
    ax.set_title('Contour Gaussian Distribution')
    plt.show()   
    
    
    fig, ax = plt.subplots()
        
    plt.plot(X[:, 0], X[:, 1], 'bx')                                                                    # Scatter de los datos
    cs = ax.contour(u, v, z, 10.**np.arange(-21, -2, 3), cmap='viridis')                                # Contour de los datos de prueba calculados
    ax.contour(u, v, z, 10, cmap='viridis')                                                             # Contour de los datos de prueba calculados
    cs.collections[0].set_label("Decision boundary")                                                    # Establecemos la leyenda
    ax.set_title('Contour Gaussian Distribution')                                                       # Establecemos titulo
    if X_anom is not None:                                                                              # Scatter plot de las anomalias
        plt.plot(X_anom[:, 0], X_anom[:, 1], 'ro', ms=8, mfc='none', mec='r')
    plt.legend()
    plt.show()                                                                                          # Visualizamos la grafica