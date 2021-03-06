import sys, os
sys.path.append(os.path.abspath(__file__).split('test')[0])

import warnings
warnings.filterwarnings("ignore")

import scipy.io as io
from pyml.unsupervised.anomaly_detection.AnomalyDetection import AnomalyDetection
from pyml.unsupervised.anomaly_detection.utils import plotData, plotDataAnom, plotDistribucion


"""

-------------------------------------------------------------------------------------------------------------------------------

                                                    EJEMPLO ORDENADORES

-------------------------------------------------------------------------------------------------------------------------------
"""

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data = io.loadmat('../../../data/ex8data1.mat')
print('Columnas: ', data.keys())

X = data['X']
X_val = data['Xval']
y_val = data['yval']


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ VISUALIZACION --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

plotData(X)


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- TRAINING -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

ad = AnomalyDetection(X)
ad.fit()


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ VISUALIZACION --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

plotDistribucion(ad, ad.X)


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ OPTIMIZACI??N ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

ad.minimizar_threshold(X_val, y_val)
print()
print("Threshold con mejor F1-score:", ad.threshold)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

X_anom, indices = ad.encontrar_anomalias(X_val)
plotDistribucion(ad, X_val, X_anom)


plotDataAnom(X_val, X_anom)


"""

-------------------------------------------------------------------------------------------------------------------------------

                                    EJEMPLO SIMPLE CON MULTIVARIATE GAUSSIAN

-------------------------------------------------------------------------------------------------------------------------------
"""


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- TRAINING -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

ad = AnomalyDetection(X, metodo='Multivariate-Gaussian')
ad.fit()


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ VISUALIZACION --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

plotDistribucion(ad, ad.X)


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ OPTIMIZACI??N ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

f1_Score = ad.minimizar_threshold(X_val, y_val)
print()
print("Threshold con mejor F1-score:", ad.threshold)
print("Mejor F1-score:", f1_Score)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

X_anom, indices = ad.encontrar_anomalias(X_val)
plotDistribucion(ad, X_val, X_anom)
plotDataAnom(X_val, X_anom)


"""

-------------------------------------------------------------------------------------------------------------------------------

                                                    EJEMPLO MAS FEATURES

-------------------------------------------------------------------------------------------------------------------------------
"""

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


data = io.loadmat('../../../data/ex8data2.mat')
print('Columnas: ', data.keys())

X = data['X']
X_val = data['Xval']
y_val = data['yval']


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- TRAINING -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

ad = AnomalyDetection(X)
ad.fit()


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ OPTIMIZACI??N ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

bestF1 = ad.minimizar_threshold(X_val, y_val)
print()
print("Threshold con mejor F1-score:", ad.threshold)
print()
print("Mejor F1-score:", bestF1)

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

X_anom, indices = ad.encontrar_anomalias(X_val)
print(X_anom.shape)

X_anom, indices = ad.encontrar_anomalias(X)
print(X_anom.shape)


"""

-------------------------------------------------------------------------------------------------------------------------------

                                    EJEMPLO SIMPLE CON MULTIVARIATE GAUSSIAN

-------------------------------------------------------------------------------------------------------------------------------
"""


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- TRAINING -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

ad = AnomalyDetection(X, metodo='Multivariate-Gaussian')
ad.fit()

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ OPTIMIZACI??N ---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

f1_Score = ad.minimizar_threshold(X_val, y_val)
print()
print("Threshold con mejor F1-score:", ad.threshold)
print("Mejor F1-score:", f1_Score)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------- TEST -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

X_anom, indices = ad.encontrar_anomalias(X_val)
print(X_anom.shape)

X_anom, indices = ad.encontrar_anomalias(X)
print(X_anom.shape)

