import sys, os
sys.path.append(os.path.abspath(__file__).split('test')[0])

import pandas as pd
import numpy as np
from pyml.supervised.LogisticRegression import LogisticRegression

"""

------------------------------------------------------------------------------------------------------------------------

                                             EJEMPLO ESTUDIANTES

------------------------------------------------------------------------------------------------------------------------

"""

data_frame = pd.read_csv("../../../data/ex2data1.txt", sep = ",", header=None)
data_frame.columns = ["examen_1", "examen_2", "admitido"]

print("-----------------------------------------")
print("             ESTUDIANTES")
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
regressor = LogisticRegression(X, y)


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- VISUALIZACION DATOS ------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


markers = np.array(['+' if admision == 1 else 'o' for admision in data_frame["admitido"].values])

color_label={
    'o': {
        'color': 'yellow',
        'label': 'No admitido'
    },
    '+': {
        'color': 'black',
        'label': 'Admitido'
    }
}

regressor.plot_datos("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- FMINFUNC -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print("Coste inicial: ", regressor.calculo_coste())
regressor.gradient_descent(0.00001, 100)
print("Coste gradiente: ", regressor.calculo_coste())
regressor.minimize()
print("Coste minimizacion: ", regressor.calculo_coste())

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------- VISUALIZACION RESULTADOS -----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

regressor.plot_resultados("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)

#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- ECUACION NORMAL ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

regressor.norm_ecuacion()
print("Coste ecuacion normal: ", regressor.calculo_coste())

regressor.plot_resultados("Admision estudiantes", "Resultado examen 1", "Resultado examen 2", markers, color_label)