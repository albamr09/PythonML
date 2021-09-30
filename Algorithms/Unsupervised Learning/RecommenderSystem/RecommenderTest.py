import scipy.io as io
import numpy as np
from RecommenderSystem import RecommenderSystem
import pandas as pd



"""

-------------------------------------------------------------------------------------------------------------------------------

                                                    EJEMPLO MOVIE RATINGS

-------------------------------------------------------------------------------------------------------------------------------
"""


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- OBTENCION DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

data = io.loadmat('../../../data/ex8_movies.mat')
print('Columnas:', data.keys())


dataParams = io.loadmat('../../../data/ex8_movieParams.mat')
print('Columnas:', dataParams.keys())

Y = data['Y']
R = data['R']
X = dataParams['X']
theta = dataParams['Theta']


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- INFORMACIÓN DATOS --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


print('\n------------------------------------------------------')
print('--------------------- INFORMACION --------------------')
print('------------------------------------------------------\n')

print('Dimension Y:', Y.shape)
print('Dimension R: ', R.shape)
print('Dimension X:', X.shape)
print('Dimension theta: ', theta.shape)


#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------- TEST DESCENSO GRADIENTE CONTENT BASED ---------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print('\n------------------------------------------------------')
print('--------------- DESCENSO GRADIENTE COBA --------------')
print('------------------------------------------------------\n')

rs = RecommenderSystem(Y, R, X=X, theta=theta, lr=0.001)
print('Coste inicial CoBa: ', rs.calculo_coste())
rs.descenso_gradiente(iter=10)
print('Coste tras descenso gradiente CoBa: ', rs.calculo_coste())


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------- TEST DESCENSO GRADIENTE COLLABORATIVE FILTERING --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print('\n------------------------------------------------------')
print('---------------- DESCENSO GRADIENTE COFI -------------')
print('------------------------------------------------------\n')

rs = RecommenderSystem(Y, R, X=X, theta=theta, optim='CoFi', lr=0.001)
print('Coste inicial CoFi: ', rs.calculo_coste())
rs.descenso_gradiente()
print('Coste tras descenso gradiente CoFi: ', rs.calculo_coste())


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ GRADIENT CHECKING ----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

print('\n------------------------------------------------------')
print('------------------ GRADIENT CHECKING -----------------')
print('------------------------------------------------------\n')

rs.gradient_checking()


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- RECOMENDACIONES Y RELACIONADOS --------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

data_movie_ids = pd.read_fwf('../../../data/movie_ids.txt', header=None)
peliculas = np.ravel(data_movie_ids.values)

num_movies = Y.shape[0]
my_ratings = np.zeros(num_movies)

my_ratings[1-1] = 4
my_ratings[98-1] = 2
my_ratings[7-1] = 3
my_ratings[12-1]= 5
my_ratings[54-1] = 4
my_ratings[64-1]= 5
my_ratings[66-1]= 3
my_ratings[69-1] = 5
my_ratings[183-1] = 4
my_ratings[226-1] = 5
my_ratings[355-1] = 5


print('\n------------------------------------------------------')
print('---------- RECOMENDACIONES Y RELACIONADOS ------------')
print('------------------------------------------------------\n')

rs = RecommenderSystem(Y, R, X=None, theta=None, optim='CoFi')
print('Coste inicial CoFi: ', rs.calculo_coste())
#rs.minimizacion(iter=1000)
print('Coste tras minimizacion CoFi: ', rs.calculo_coste())

rs.anadir_usuario(0, my_ratings, entrenar=True, iter=1000)
print('Coste tras anadir usuario CoFi: ', rs.calculo_coste())

rs.items_relacionados(225, verbose=True, items_ids=peliculas)
rs.items_relacionados(354, verbose=True, items_ids=peliculas)
rs.items_relacionados(68, verbose=True, items_ids=peliculas)
rs.items_relacionados(64, verbose=True, items_ids=peliculas)
rs.items_relacionados(11, verbose=True, items_ids=peliculas)
rs.recomendaciones_usuario(0, verbose=True, items_ids=peliculas)
