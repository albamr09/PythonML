# PythonML

Implementation from scratch of some machine learning algorithms.

## Modules

- **supervised**
  - linear_regression
    - _LinearRegression_: Simple linear regressor
    - _LinearRegressionV2_: Linear regressor with more features like regularization, feature normalization, etc.
  - SVM
    - _SVM_: My own SVM model (Still working on it)
    - _SVM2_: Coursera based SVM model
  - _LogisticRegresssion_
  - _NeuralNetwork_
- **unsupervised**
  - anomaly_detection
    - _AnomalyDetection_
  - clustering
    - _KMeans_
    - _PCA_
  - _RecommenderSystem_

---

## Dependencies

The algorithms were implemented using `Python 3.8.10`.

| Name             |      Version      |
| ---------------- | :---------------: |
| **matplotlib**   |  >=3.3.3,< 3.4.0  |
| **numpy**        | >=1.19.5,< 1.20.0 |
| **scipy**        |  >=1.5.4,< 1.6.0  |
| **scikit-learn** | >=0.24.1,< 0.25.0 |
| **pandas**       |  >=1.1.4,< 1.2.0  |

Install with

```console
$ pip3 install -r requirements.txt
```

Where:

```console
$ pip3 --version
pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)
```

---

When testing the `SpamClassifier.py` the following dependency is needed:

- regex 2.5.91
