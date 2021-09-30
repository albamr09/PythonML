import sys, os
sys.path.append(os.path.abspath(__file__).split('test')[0])

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyml.supervised.linear_regression.LinearRegression import LinearRegression


"""

------------------------------------------------------------------------------------------------------------------------

                                                    CASAS

------------------------------------------------------------------------------------------------------------------------

"""


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data = pd.read_csv('../../../data/ex1data2.txt', sep=",", header=None)         #Leer datos
data.columns = ["size", "n_bedrooms", "price"]


print(data.head())                                                          #Imprimimos los datos
print(data.describe())

num_col = data.shape[0]
num_filas = data.shape[1]

X = np.matrix([np.ones(num_col), data['size'], data['n_bedrooms']])
y = np.matrix(data['price'])


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- FEATURE NORMALIZATION -----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


multi_linear_regression = LinearRegression(X, y)
multi_linear_regression.aplicar_feature_normalization(include_y=True)

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------- VISUALIZACION ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


fig = plt.figure()
ax = fig.gca(projection='3d')

xline = multi_linear_regression.X[1]
yline = multi_linear_regression.X[2]
zline = multi_linear_regression.y

ax.scatter(xline, yline, zline, 'gray')
plt.title("Dataset")
plt.xlabel("Tamano")
plt.ylabel("Numero camas")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------




print("\n--------------------------------------------")
print("    RESULTADOS CON FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

multi_linear_regression = LinearRegression(X, y)
multi_linear_regression.aplicar_feature_normalization()

print("Coste inicial: ", multi_linear_regression.calcular_coste())
multi_linear_regression.minimizacion()
print("Coste final: ", multi_linear_regression.calcular_coste())

print("\n--------------------------------------------")
print("    RESULTADOS SIN FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

multi_linear_regression = LinearRegression(X, y)

print("Coste inicial: ", multi_linear_regression.calcular_coste())
multi_linear_regression.minimizacion()
print("Coste final: ", multi_linear_regression.calcular_coste())



"""

------------------------------------------------------------------------------------------------------------------------

                                                EJEMPLO REAL STATE

------------------------------------------------------------------------------------------------------------------------

"""


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data = pd.read_csv("../../../data/50_Startups.csv", sep=",")

print("\n--------------------------------------------")
print("          REAL STATE        ")
print("--------------------------------------------\n")

print(data.head())
print(data.describe())

data_encoded = pd.get_dummies(data)
num_filas = data_encoded.values.shape[0]
num_col = data_encoded.values.shape[1]

X = np.matrix([np.ones(num_filas)])
X = np.concatenate((X, data_encoded.loc[:, data_encoded.columns != 'Profit'].values.T))
y = np.matrix(data_encoded['Profit'].values)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print("\n--------------------------------------------")
print("    RESULTADOS SIN FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

multi_linear_regression = LinearRegression(X, y)

print("Coste inicial: ", multi_linear_regression.calcular_coste())
multi_linear_regression.minimizacion()
print("Coste final: ", multi_linear_regression.calcular_coste())


print("\n--------------------------------------------")
print("    RESULTADOS CON FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

multi_linear_regression = LinearRegression(X, y)
multi_linear_regression.aplicar_feature_normalization(exclude=[4, 6], include_y=True)

print("Coste inicial: ", multi_linear_regression.calcular_coste())
multi_linear_regression.minimizacion()
print("Coste final: ", multi_linear_regression.calcular_coste())






