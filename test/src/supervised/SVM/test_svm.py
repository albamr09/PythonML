import sys, os
sys.path.append(os.path.abspath(__file__).split('test')[0])

import scipy.io as io
import numpy as np

from pyml.supervised.SVM.SVM import SVM


data = io.loadmat('../../../data/ex6data1.mat')

y = data['y']
X = data['X']

y = np.matrix(y, dtype=int)

svm = SVM(X, y, axis=1, lr=0.01, C=10)
svm.minimizar()
print(svm.accuracy())
