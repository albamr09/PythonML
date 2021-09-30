import pandas as pd
import numpy as np
from LinearRegression import LinearRegression


"""

------------------------------------------------------------------------------------------------------------------------

                                                EJEMPLO BENEFICIO

------------------------------------------------------------------------------------------------------------------------

"""

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

data = pd.read_csv('../../../data/ex1data1.txt', sep=",", header=None)
data.columns = ["population", "profit"]


print("\n--------------------------------------------")
print("    PRECIOS DE CASAS         ")
print("--------------------------------------------\n")

print(data.head())
print(data.describe())

num_col = data.shape[0]
num_filas = data.shape[1]

X = np.matrix([np.ones(num_col), data['population']])
y = np.matrix(data['profit'])

simple_linear_regressor = LinearRegression(X, y, split=0.2)


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------ TEST Y VISUALIZACION -------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


simple_linear_regressor.gradient_descent(0.01, 1000)
simple_linear_regressor.plot_regression("Beneficios train gradient", "Poblacion", "Profit")
simple_linear_regressor.plot_regression_test("Beneficios test gradient", "Poblacion", "Profit")

simple_linear_regressor.minimizacion()
simple_linear_regressor.plot_regression("Beneficios train minimizacion", "Poblacion", "Profit")
simple_linear_regressor.plot_regression_test("Beneficios test minimizacion", "Poblacion", "Profit")


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
print("    SALARIOS        ")
print("--------------------------------------------\n")

print(data_salarios.head())
print(data_salarios.describe())

num_filas = data_salarios.values.shape[0]
num_col = data_salarios.values.shape[1]

X_salarios = np.matrix([np.ones(num_filas), np.ravel(data_salarios.iloc[:, :-1].values)])
y_salarios = np.matrix(data_salarios.iloc[:, -1].values)

simple_linear_regressor = LinearRegression(X_salarios, y_salarios, split=0.2)


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------ TEST Y VISUALIZACION -------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


simple_linear_regressor.gradient_descent(0.05, 1000)
simple_linear_regressor.plot_regression("Salarios train gradient", "Experiencia", "Salario")
simple_linear_regressor.plot_regression_test("Salarios test gradient", "Experiencia", "Salario")

simple_linear_regressor.minimizacion()
simple_linear_regressor.plot_regression("Salarios train minimizacion", "Experiencia", "Salario")
simple_linear_regressor.plot_regression_test("Salarios test minimizacion", "Experiencia", "Salario")




