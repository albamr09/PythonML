import sys, os
sys.path.append(os.path.abspath(__file__).split('test')[0])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from pyml.supervised.NeuralNetwork import NeuralNetwork as NN


"""

------------------------------------------------------------------------------------------------------------------------

                                                PLOT DATOS

------------------------------------------------------------------------------------------------------------------------

"""

def displayData(X):

    m, n = X.shape
    example_width = int(np.round(np.sqrt(n)))

    fig, ax_array = plt.subplots(10, 10, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')
    plt.show()


"""

------------------------------------------------------------------------------------------------------------------------

                                                DE LISTA A ARRAY

------------------------------------------------------------------------------------------------------------------------

"""


def unroll_input(input):
        theta_ravel = []                                # Creamos la lista que almacenará las matrices tras flatten
        for theta_element in input:
            theta_ravel.append(np.ravel(theta_element)) # Hacer que la matriz sea un vector y almacenarlo en lista temporal
        return np.concatenate(theta_ravel)              # Hacer que la lista temporal sea un solo vector



"""

------------------------------------------------------------------------------------------------------------------------

                                                EJEMPLO NUMEROS

------------------------------------------------------------------------------------------------------------------------

"""


data = io.loadmat('../../../data/ex3weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']
print("Dimensiones theta 1:", theta1.shape)
print("Dimensiones theta 2:", theta2.shape)


data = io.loadmat('../../../data/ex3data1.mat')
data = pd.DataFrame(np.hstack((data['X'], data['y'])))
print(data.info())
print(data.head())
print(data.describe())


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

m, n = data.shape

X = np.array(data.iloc[:, 0:n-1]).T
y = np.array(data.iloc[:, -1], ndmin=2)

rand_indices = np.random.choice(m, 100, replace=False)
sel = X.T[rand_indices, :]
displayData(sel)

X = np.concatenate((np.matrix(np.ones(m)), X))

#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- FORWARD PROPAGATION ------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ FIRST LAYER ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

z = theta1.dot(X)                                                                             # Calculamos la entrada a la funcion sigmoid: theta*X
a = 1 / (1 + np.exp(-z))                                                                      # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))
a = np.concatenate((np.matrix(np.ones(m)), a))                                                # Añadimos una fila de unos


#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------- SECOND LAYER ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


z = theta2.dot(a)                                                                             # Calculamos la entrada a la funcion sigmoid: theta*X
a = 1 / (1 + np.exp(-z))                                                                      # Funcion sigmoid: 1 / (1 + e^(-sum(theta*X)))

p = np.argmax(a, axis = 0)                                                                    # Obtenemos el indice del elemento de mayor valor
y = np.array(y, dtype=int)                                                                    # Casteamos array a int
p = p + 1                                                                                     # Aumentamos el valor del indice ya que las categorias son de 1 - 10
print('Precision del modelo: {:.1f}%'.format(np.mean(p == y) * 100))                          # Calculamos cuantos valores coinciden con el valor a predecir


"""

------------------------------------------------------------------------------------------------------------------------

                                                EJEMPLO NUMEROS

------------------------------------------------------------------------------------------------------------------------

"""

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

data = io.loadmat('../../../data/ex4data1.mat')
data = pd.DataFrame(np.hstack((data['X'], data['y'])))

X = np.array(data.iloc[:, 0:n-1])
y = np.array(data.iloc[:, -1], ndmin=2)


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ CREACION NN ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

nn = NN(X, y, axis=1)
nn.anadir_capa((25, 401))
nn.anadir_capa((10, 26))


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------- PRUEBA CON PESOS PREDETERMINADOS ------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

nn.theta[0] = theta1
nn.theta[1] = theta2
print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo()))  
print('Coste sin regularizacion:', nn.calculo_coste()) 
print('Coste sin regularizacion unrolled:', nn.calculo_coste(theta=unroll_input(nn.theta), unrolled=True))


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------- CREACION NN CON REGULARIZACION -------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

nn = NN(X, y, axis=1, reg=True, reg_par=1, random=True, split=0.2)
nn.anadir_capa((25, 401))
nn.anadir_capa((10, 26))


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print("\n---------------------------------------------------------------------")
print("------------------- TEST CON REGULARIZACION -------------------------")
print("---------------------------------------------------------------------\n")

print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo()))  
print('Coste con regularizacion:', nn.calculo_coste()) 
print('Coste con regularizacion unrolled:', nn.calculo_coste(theta=unroll_input(nn.theta), unrolled=True))

nn.back_propagation()
nn.gradient_checking()
nn.minimizacion(iter=100)

print('Coste tras minimizacion con regularizacion:', nn.calculo_coste())
print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo())) 


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------- CREACION NN SIN REGULARIZACION -------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


nn = NN(X, y, axis=1, random=True, split=0.2)
nn.anadir_capa((25, 401))
nn.anadir_capa((10, 26))


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("\n---------------------------------------------------------------------")
print("------------------- TEST SIN REGULARIZACION -------------------------")
print("---------------------------------------------------------------------\n")

print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo()))  
print('Coste con regularizacion:', nn.calculo_coste()) 
print('Coste con regularizacion unrolled:', nn.calculo_coste(theta=unroll_input(nn.theta), unrolled=True))

nn.back_propagation()
nn.gradient_checking()
nn.minimizacion(iter=100)

print('Coste tras minimizacion con regularizacion:', nn.calculo_coste())
print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo())) 


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- MODIFICACION DE DATOS -----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


y = pd.Series(y[0])
y = y.map({1: 'Uno', 
       2: 'Dos',
       3: 'Tres',
       4: 'Cuatro',
       5: 'Cinco',
       6: 'Seis',
       7: 'Siete',
       8: 'Ocho',
       9: 'Nueve',
       10: 'Diez'})
y = y.to_numpy()


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------- CREACION NN CON REGULARIZACION -------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

nn = NN(X, y, axis=1, reg=True, reg_par=1, random=True, split=0.2)
nn.anadir_capa((25, 401))
nn.anadir_capa((10, 26))


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print("\n---------------------------------------------------------------------")
print("------------- TEST CON REGULARIZACION Y CATEGORIAS ------------------")
print("---------------------------------------------------------------------\n")

print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo()))  
print('Coste con regularizacion:', nn.calculo_coste()) 
print('Coste con regularizacion unrolled:', nn.calculo_coste(theta=unroll_input(nn.theta), unrolled=True))

nn.back_propagation()
nn.gradient_checking()
nn.minimizacion(iter=100)

print('Coste tras minimizacion con regularizacion:', nn.calculo_coste())
print('Precision del modelo: {:.1f}%'.format(nn.precision_modelo())) 