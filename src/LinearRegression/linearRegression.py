import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class linearRegression :
    def fit(self, X_in, Y_in) :
        return np.linalg.inv(X_in.T @ X_in) @ X_in.T @ Y_in

df = pd.read_csv("../../datasets/winequality-red.csv")
X_Raw = df.iloc[:, 0:11].to_numpy()
Y = df.iloc[:, 11].to_numpy()

n, m = X_Raw.shape
appendCol =  np.ones((n,1))
X = np.hstack((X_Raw, appendCol))

trainSize = int(n*.8)
X_Train = X[0:trainSize,:]
Y_Train = Y[0:trainSize]
X_Test = X[trainSize:, :]
Y_Test = Y[trainSize:]

X_Trans = np.matrix.transpose(X_Train)

linearReg = linearRegression()
w = linearReg.fit( X_Train, Y_Train)

Y_Predicted = X_Test @ w
error = 1/Y_Test.shape[0] * np.linalg.norm(Y_Test - Y_Predicted) ** 2

print(error)