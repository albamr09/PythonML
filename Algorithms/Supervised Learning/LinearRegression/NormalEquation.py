import pandas as pd
import numpy as np
from LinearRegression import LinearRegression


"""

------------------------------------------------------------------------------------------------------------------------

                                            SIMPLE LINEAR REGRESSION

------------------------------------------------------------------------------------------------------------------------

"""


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data = pd.read_csv('../../../data/ex1data1.txt', sep=",", header=None)             #Cargamos los datos
data.columns = ["population", "profit"]

print("\n--------------------------------------------")
print("--------------------------------------------")
print("         SIMPLE LINEAR REGRESSION           ")
print("--------------------------------------------")
print("--------------------------------------------\n")

print(data.head())                                                              #Imprimimos los datos
print(data.describe())

num_col = data.shape[0]                                                         #Numero de columnas: numero de ejemplos
num_filas = data.shape[1]                                                       #Numero de filas: numero de features

X = np.matrix([np.ones(num_col), data['population']]).T                         #Cada fila es un ejemplo, cada columna es un feature del ejemplo
y = np.matrix(data['profit']).T                                                 #Vector columna


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


norm_linear_regression = LinearRegression(X, y, axis=1)
norm_linear_regression.norm_equation()

print("\n--------------------------------------------")
print("               RESULTADOS                  ")
print("--------------------------------------------\n")

print("\nMinimizacion parámetros theta: ", norm_linear_regression.theta)
print("Dimension theta: ", norm_linear_regression.theta.shape)
print("Theta como array: ", np.ravel(norm_linear_regression.theta))
print("Prueba prediccion de: ", 6.1101, " con nuestro modelo de Linear Regression: ", norm_linear_regression.prediccion(np.matrix([1, 6.1101]).T))

norm_linear_regression.plot_regression("Plot sin division", "Poblacion", "Beneficio")


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- DIVISION Y TEST --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


norm_linear_regression = LinearRegression(X, y, axis=1, split=0.2)
norm_linear_regression.norm_equation()
norm_linear_regression.plot_regression("Training set", "Poblacion", "Beneficio")
norm_linear_regression.plot_regression_test("Test set", "Poblacion", "Beneficio")


"""

------------------------------------------------------------------------------------------------------------------------

                                                EJEMPLO SALARIOS

------------------------------------------------------------------------------------------------------------------------

"""

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data_salarios = pd.read_csv("../../../data/Salary_Data.csv", sep=",")

print("\n--------------------------------------------")
print("                 SALARIOS        ")
print("--------------------------------------------\n")

print(data_salarios.head())
print(data_salarios.describe())

num_filas = data_salarios.values.shape[0]
num_col = data_salarios.values.shape[1]

X_salarios = np.matrix([np.ones(num_filas), np.ravel(data_salarios.iloc[:, :-1].values)]).T
y_salarios = np.matrix(data_salarios.iloc[:, -1].values).T


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- DIVISION Y TEST --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


norm_linear_regression = LinearRegression(X_salarios, y_salarios, axis=1, split=0.2)
norm_linear_regression.norm_equation()
norm_linear_regression.plot_regression("Training set", "Experiencia", "Salario")
norm_linear_regression.plot_regression_test("Test set", "Experiencia", "Salario")



"""

------------------------------------------------------------------------------------------------------------------------

                                                    MULTIPLE LINEAR REGRESSION

------------------------------------------------------------------------------------------------------------------------

"""


print("\n--------------------------------------------")
print("--------------------------------------------")
print("         MULTIPLE LINEAR REGRESSION           ")
print("--------------------------------------------")
print("--------------------------------------------\n")


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data = pd.read_csv('../../../data/ex1data2.txt', sep=",", header=None)          #Cargamos los datos
data.columns = ["size", "n_bedrooms", "price"]

print(data.head())                                                              #Imprimimos los datos
print(data.describe())

num_col = data.shape[0]                                                         #Numero de columnas: numero de ejemplos
num_filas = data.shape[1]                                                       #Numero de filas: numero de features

X = np.matrix([np.ones(num_col), data['size'], data['n_bedrooms']]).T           #Cada fila es un ejemplo, cada columna es un feature del ejemplo
y = np.matrix(data['price']).T                                                  #Vector columna


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print("\n--------------------------------------------")
print("    RESULTADOS CON FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

norm_linear_regression = LinearRegression(X, y, axis=1)
norm_linear_regression.aplicar_feature_normalization(include_y=True)
norm_linear_regression.norm_equation()

print("\nMinimizacion parámetros theta: ", norm_linear_regression.theta)
print("Dimension theta: ", norm_linear_regression.theta.shape)
print("Theta como array: ", np.ravel(norm_linear_regression.theta))
print("Prueba prediccion de: ", 2104, " size y ", 3, " numero de camas, con nuestro modelo de Linear Regression: "
      , norm_linear_regression.prediccion(np.matrix([np.ones(1), np.matrix([2104]), np.matrix([3])])), ", valor real: ", y[0])


print("\n--------------------------------------------")
print("    RESULTADOS SIN FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

norm_linear_regression = LinearRegression(X, y, axis=1)
norm_linear_regression.norm_equation()


print("\nMinimizacion parámetros theta: ", norm_linear_regression.theta)
print("Dimension theta: ", norm_linear_regression.theta.shape)
print("Theta como array: ", np.ravel(norm_linear_regression.theta))
print("Prueba prediccion de: ", 2104, " size y ", 3, " numero de camas, con nuestro modelo de Linear Regression: "
      , norm_linear_regression.prediccion(np.matrix([np.ones(1), np.matrix([2104]), np.matrix([3])])), ", valor real: ", y[0])



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
print("               REAL STATE        ")
print("--------------------------------------------\n")

print(data.head())
print(data.describe())

data_encoded = pd.get_dummies(data)
num_filas = data_encoded.values.shape[0]
num_col = data_encoded.values.shape[1]

X = np.matrix([np.ones(num_filas)])
X = np.concatenate((X, data_encoded.loc[:, data_encoded.columns != 'Profit'].values.T)).T
y = np.matrix(data_encoded['Profit'].values).T


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print("\n--------------------------------------------")
print("    RESULTADOS SIN FEATURE NORMALIZATION          ")
print("--------------------------------------------\n")

norm_linear_regression = LinearRegression(X, y, axis=1)
norm_linear_regression.norm_equation()


print("\nMinimizacion parámetros theta: ", norm_linear_regression.theta)
print("Dimension theta: ", norm_linear_regression.theta.shape)
print("Theta como array: ", np.ravel(norm_linear_regression.theta))
print("Prueba prediccion: ", norm_linear_regression.prediccion(X[0, :].T), ", valor real: ", y[0])


