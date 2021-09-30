import sys, os
sys.path.append(os.path.abspath(__file__).split('test')[0])

import pandas as pd
import numpy as np
from pyml.supervised.LogisticRegression import LogisticRegression

"""

------------------------------------------------------------------------------------------------------------------------

                                             EJEMPLO MICROCHIPS

------------------------------------------------------------------------------------------------------------------------

"""

data_frame = pd.read_csv("../../../data/ex2data2.txt", sep = ",", header=None)
data_frame.columns = ["test_1", "test_2", "aceptado"]

print("-----------------------------------------")
print("             MICROCHIPS")
print("-----------------------------------------")
print(data_frame.head())
print("-----------------------------------------")
print(data_frame.describe())
print("-----------------------------------------")


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


X = np.matrix(data_frame.values[:, :-1]).T
y = np.matrix(data_frame.values[:, -1])

regressor = LogisticRegression(X, y, reg=True, reg_par=1)

#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- VISUALIZACION DATOS ------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


markers = np.array(['+' if admision == 1 else 'o' for admision in data_frame["aceptado"].values])

color_label={
    'o': {
        'color': 'yellow',
        'label': 'No aceptado'
    },
    '+': {
        'color': 'black',
        'label': 'Aceptado'
    }
}

regressor.plot_datos("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------- VISUALIZACION RESULTADOS -----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("Coste inicial: ", regressor.calculo_coste())
regressor.gradient_descent(1, 100)
print("Coste gradiente: ", regressor.calculo_coste())

regressor.plot_resultados("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- FMINFUNC -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

regressor.minimize()
print("Coste minimizacion: ", regressor.calculo_coste())

regressor.plot_resultados("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)


#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- ECUACION NORMAL ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

regressor.norm_ecuacion()
print("Coste ecuacion normal: ", regressor.calculo_coste())

regressor.plot_resultados("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)