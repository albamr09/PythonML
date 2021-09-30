import numpy as np
import matplotlib.pyplot as plt


"""

------------------------------------------------------------------------------------------------------------------------

                                                CLUSTERING TEST

------------------------------------------------------------------------------------------------------------------------

"""


"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que hace una gráfica con los datos y la sucesión de centroids.

            -- X: datos.
            -- indices: lista que indica a que cluster pertenece cada ejemplo.
            -- lista_centroids: lista que almacena el valor de los centroids en cada iteracion.

    ------------------------------------------------------------------------------------------------------------------------

"""

def plot_resultados(X, indices, lista_centroids):
    
    markers = np.array(['o' if indice == 0 else '+' if indice == 1 else 'D' for indice in indices]) # Cada cluster tiene un marker

    color_label={                                                                                   # Definimos legend y color de cada cluster
        'o': {
            'color': 'blue',
            'label': 'Clase azul'
        },
        '+': {
            'color': 'red',
            'label': 'Clase roja'
        },
        'D': {
            'color': 'green',
            'label': 'Clase verde'
        }
    }


    fig, ax = plt.subplots()                                                                                # Creamos un subplot

    for marker in np.unique(markers):                                                                       # Para cada categoria hacemos un scatter
        ax.scatter(X[:, 0][markers == marker], X[:, 1][markers == marker],
                        marker=marker,
                        color=color_label[marker]['color'], label=color_label[marker]['label'])

    plt.scatter(lista_centroids[:, 0, 0], lista_centroids[:, 0, 1], marker='X', color='black')              # Hacemos una gráfica de la posicion de los centroids en cada iteracion
    plt.plot(lista_centroids[:, 0, 0], lista_centroids[:, 0, 1], 'k')                                       # Hacemos una gráfica de la alteracion de la posicion
    plt.scatter(lista_centroids[:, 1, 0], lista_centroids[:, 1, 1], marker='X', color='black')              # Hacemos una gráfica de la posicion de los centroids en cada iteracion
    plt.plot(lista_centroids[:, 1, 0], lista_centroids[:, 1, 1], 'k')                                       # Hacemos una gráfica de la alteracion de la posicion
    plt.scatter(lista_centroids[:, 2, 0], lista_centroids[:, 2, 1], marker='X', color='black')              # Hacemos una gráfica de la posicion de los centroids en cada iteracion
    plt.plot(lista_centroids[:, 2, 0], lista_centroids[:, 2, 1], 'k')                                       # Hacemos una gráfica de la alteracion de la posicion
    plt.legend()
    plt.show()
    

"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que crea un grid de dos imagenes.

            -- original: imagen original.
            -- comprimida: imagen comprimida.

    ------------------------------------------------------------------------------------------------------------------------

"""

def ver_imagen(original, comprimida):
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))        # Creamos un grid de una fila y dos columnas con figuras de 8x4
    ax[0].imshow(original)                              # Mostramos por pantalla la imagen original
    ax[0].set_title('Original')
    ax[0].grid(False)

    ax[1].imshow(comprimida)                            # Mostramos por pantalla la imagen comprimida
    ax[1].set_title('Comprimida')
    ax[1].grid(False)

    plt.show()
    
    
    
"""

------------------------------------------------------------------------------------------------------------------------

                                                    PCA TEST
                                             
------------------------------------------------------------------------------------------------------------------------

"""


"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que permita visualizar una gráfica de tipo scatter de los datos en 2D.

            -- X: datos -> filas: ejemplos , columnas: features. Por lo tanto la primera columna
                son las X y las segunda las y.

    ------------------------------------------------------------------------------------------------------------------------

"""
    
def plot_data(X):
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='b')    # Hacemos la gráfica de los datos.
    plt.show()
    

"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que permita visualizar un grid de imagenes.

            -- imagenes: lista de imagenes.
            -- filas: numero de filas del grid.
            -- columnas: numero de columnas del grid.
            -- size: tamaño de las imagenes.

    ------------------------------------------------------------------------------------------------------------------------

"""

def ver_imagenes(imagenes, filas, columnas, size):
    
    fig, subplots = plt.subplots(filas, columnas, figsize=size)     # Creamos el grid con las caracteristicas especificadas
    fig.subplots_adjust(wspace=0.025, hspace=0.025)                 # Hacemos los espacion entre imágenes más pequeños.
    
    subplots = subplots.ravel()                                     # Hacemos unroll de los subplots.
    
    for indice, ax in enumerate(subplots):                          # Por cada subplot obtenemos el indice del mismo y el objeto en si
        ax.imshow(imagenes[indice], cmap='gray')                    # Mostramos por pantalla la imagen correspondiente
        ax.axis('off')                                              # No mostrar ejes
        
    plt.show()    


"""

    ------------------------------------------------------------------------------------------------------------------------

            Funcion que permita visualizar la reduccion de 3D a 2D.

            -- X: datos originales.
            -- X_comprimido: proyeccion de X sobre z.
            -- z: X tras reducirse su dimension.

    ------------------------------------------------------------------------------------------------------------------------

"""

def ver_3D_a_2D(X_comprimido, X, z):
    
    m, c = X_comprimido.shape                           # Guardamos las dimensiones de los datos             
    indices_random = np.random.permutation(m)           # Obtenemos indice aleatorios a partir del numero total de datos
    seleccion_aleatoria = X[indices_random[:1000]]      # Almacenamos los datos correspondientes a los 1000 primeros indice aleatorios.
    colores = X_comprimido[indices_random[:1000]]       # Almacenamos los colores a los que se ha reducido los datos originales.

    fig = plt.figure()                                  # Creamos un grid
    ax = fig.add_subplot(111, projection='3d')          # Creamos la proyeccion 3D

    ax.scatter(seleccion_aleatoria[:, 0],               # Hacemos la grafica de los datos y asignamos los colores
               seleccion_aleatoria[:, 1], 
               seleccion_aleatoria[:, 2], 
               c=colores)
    
    plt.show()
    
    seleccion_aleatoria = z[:, indices_random[:1000]]   # Almacenamos los datos de menor dimension correspondientes a los 1000 primeros indice aleatorios.

    plt.scatter(seleccion_aleatoria[0, :],              # Hacemos una grafica de los datos en 2D y asignamos los colores
                seleccion_aleatoria[1, :], 
                c=colores)
    
    plt.show()