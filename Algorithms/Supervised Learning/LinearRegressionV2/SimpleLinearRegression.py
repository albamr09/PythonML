import pandas as pd
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
from LinearRegression import LinearRegression


"""

------------------------------------------------------------------------------------------------------------------------

                                                EJEMPLO NIVEL AGUA

------------------------------------------------------------------------------------------------------------------------

"""

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

data = io.loadmat('../../../data/ex5data1.mat')


X = data['X']
y = data['y']
X_test = data['Xtest']
y_test = data['ytest']
X_val = data['Xval']
y_val = data['yval']


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ VISUALIZACION --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

plt.scatter(X, y, marker='x')
plt.title("Ver datos")
plt.xlabel("Cambio en el nivel del agua")
plt.ylabel("Agua que sale")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------- TEST ------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print("\n--------------------------------------------")
print("                    TEST        ")
print("--------------------------------------------\n")

reg = LinearRegression(X, y, axis=1, test=(X_test, y_test), val=(X_val, y_val), reg=True, reg_par=1)

print("Coste sin entrenar", reg.calcular_coste())


"""

------------------------------------------------------------------------------------------------------------------------

                                        EJEMPLO NIVEL AGUA SIN DIVIDIR

------------------------------------------------------------------------------------------------------------------------

"""


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


X = np.vstack((X,X_val, X_test))
y = np.vstack((y,y_val, y_test))


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------- TEST CON REGULARIZACION = 1 --------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("\n--------------------------------------------")
print("          TEST CON REGULARIZACION = 1        ")
print("--------------------------------------------\n")

reg = LinearRegression(X, y, axis=1, split=0.38, val_split=0.63, reg=True, reg_par=1)

print("Coste sin entrenar", reg.calcular_coste())

reg.minimizacion()

print("Coste tras entrenar", reg.calcular_coste())

reg.plot_regression("Fit Modelo", "Cambio en el nivel del agua", "Agua que sale")

reg.learning_curve()


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------- TEST CON REGULARIZACION = 0 --------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("\n--------------------------------------------")
print("         TEST CON REGULARIZACION = 0        ")
print("--------------------------------------------\n")

reg = LinearRegression(X, y, axis=1, split=0.38, val_split=0.63, reg=True, reg_par=0, grado=8, feature_norm=True)

print("Coste sin entrenar poly", reg.calcular_coste())

reg.minimizacion()

print("Coste tras entrenar poly", reg.calcular_coste())

reg.plot_poly_regression("Fit Modelo Poly reg_par = 0", "Cambio en el nivel del agua", "Agua que sale")

reg.learning_curve()


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------- TEST POLYNOMIAL, FN, CON REGULARIZACION = 1 -------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("\n------------------------------------------------------")
print("    TEST POLYNOMIAL, FN, CON REGULARIZACION = 1        ")
print("--------------------------------------------------------\n")

reg = LinearRegression(X, y, axis=1, split=0.38, val_split=0.63, reg=True, reg_par=1, grado=8, feature_norm=True)

print("Coste sin entrenar poly", reg.calcular_coste())

reg.minimizacion()

print("Coste tras entrenar poly", reg.calcular_coste())


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------ TEST POLYNOMIAL, FN, CON REGULARIZACION = 100 ------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("\n--------------------------------------------------------")
print("    TEST POLYNOMIAL, FN, CON REGULARIZACION = 100        ")
print("--------------------------------------------------------\n")

reg = LinearRegression(X, y, axis=1, split=0.38, val_split=0.63, reg=True, reg_par=100, grado=8, feature_norm=True)

print("Coste sin entrenar poly", reg.calcular_coste())

reg.minimizacion()

print("Coste tras entrenar poly", reg.calcular_coste())

reg.plot_poly_regression("Fit Modelo Poly reg_par = 100", "Cambio en el nivel del agua", "Agua que sale")

reg.learning_curve()

reg.mejor_reg_par([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 10])


#-----------------------------------------------------------------------------------------------------------------------
#----------------------------- TEST POLYNOMIAL, FN, CON REGULARIZACION = 0.01 ------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("\n---------------------------------------------------------")
print("    TEST POLYNOMIAL, FN, CON REGULARIZACION = 0.01        ")
print("----------------------------------------------------------\n")

reg = LinearRegression(X, y, axis=1, split=0.38, val_split=0.63, reg=True, reg_par=0.01, grado=8, feature_norm=True)

print("Coste sin entrenar poly", reg.calcular_coste())

reg.minimizacion()

print("Coste tras entrenar poly", reg.calcular_coste())

reg.plot_poly_regression("Fit Modelo Poly reg_par = 100", "Cambio en el nivel del agua", "Agua que sale")

reg.learning_curve()

reg.mejor_reg_par([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 10])

