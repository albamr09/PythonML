import pandas as pd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from LogisticRegression import LogisticRegression, MultiLogisticRegression


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

                                             EJEMPLO NUMEROS

------------------------------------------------------------------------------------------------------------------------

"""


data = io.loadmat('../../../data/ex3data1.mat')
data = pd.DataFrame(np.hstack((data['X'], data['y'])))
print(data.info())
print(data.head())
print(data.describe())


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

m, n = data.shape

X = np.array(data.iloc[:, 0:n-1])
y = np.array(data.iloc[:, -1], ndmin=2)

#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- VISUALIZACION DATOS ------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
displayData(sel)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------- VISUALIZACION RESULTADOS -----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


mulregression = MultiLogisticRegression(X, y, axis=1, reg=True, reg_par=0.1, categorias=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
print("Coste inicial:", mulregression.calculo_coste())
mulregression.gradient_descent(0.1, 100)
print("Coste tras descenso del gradiente:", mulregression.calculo_coste())
mulregression.minimize()
print("Coste tras minimizacion:", mulregression.calculo_coste())
print("Prediccion de nuestro modelo:", mulregression.prediccion(mulregression.X[:, 0]), "\nResultado real:", y[0, 0])
print("Precision de nuestro modelo: {precision:.2f}%".format(precision=(mulregression.precision()*100)))


