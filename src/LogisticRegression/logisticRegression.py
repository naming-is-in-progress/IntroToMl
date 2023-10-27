import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class logisticRegression :
    gdIterter : int
    learningRate : int
    fitLoss : np.ndarray(0, dtype=float)
    def __init__(self, iter, stepSize) :
        self.gdIter = iter
        self.learningRate = stepSize
    
    def sigmoid(self, x) :
        return 1 / (1 + np.exp(x * -1))

    def cost(self, Xin, Yin, w):
        n, m = Xin.shape
        loss = 0.0
        for i in range(n) :
            loss += np.log(1/self.sigmoid(Yin[i] * w.T @ Xin[i]))
        return loss/n
    
    def gradient_descent(self, X_in, Y_in, w) :
        n, m = X_in.shape
        g = np.zeros(m)
        for i in range (n) :
            denominator = self.sigmoid(Y_in[i] * w.T @ X_in[i])
            numerator = -1 * Y_in[i] * np.exp(-1 * Y_in[i] * w.T @ X_in[i])
            g += X_in[i] * (numerator/denominator)
        for i in range (m) :
            if g[i] > 1000:
                g /= 1000
                break
        return g/n
    
    def fit (self, X_in, Y_in) :
        self.fitLoss = np.ndarray(0, dtype=float)
        w = np.random.rand(X_in.shape[1])
        for _ in range (100000) :
            w -=  self.learningRate * self.gradient_descent(X_in, Y_in, w)
            np.append(self.fitLoss, self.cost(X_in, Y_in, w))
        return w
    
print(os.getcwd())
df = pd.read_csv("../../datasets/penguins.csv")
df.dropna(inplace=True)
n, m = df.shape
df['island'] = df['island'].astype('category')
df['species'] = df['species'].astype('category')
df['sex'] = df['sex'].astype('category')
catCodeCols = df.select_dtypes(['category']).columns
df[catCodeCols] = df[catCodeCols].apply(lambda x : x.cat.codes)

nonCatCols = df.select_dtypes(['float']).columns
df[nonCatCols] = df[nonCatCols].apply(lambda x: (x - x.min())/(x.max()-x.min()))

X_Raw = df.iloc[:, 0:m-2].to_numpy()
appendCol =  np.ones((X_Raw.shape[0],1))

X = np.hstack((X_Raw, appendCol))

Y = df.iloc[:,m-2].to_numpy()
Y = np.where(Y == 0, -1, Y)
    
trainSize = int(Y.shape[0]*.8)
X_Train = X[0:trainSize,:]
Y_Train = Y[0:trainSize]
X_Test = X[trainSize:, :]
Y_Test = Y[trainSize:]

logit = logisticRegression(100000, .01)
#print(logit.gradient_descent(X_Train, Y_Train, np.ones(X_Train.shape[1])))
w = logit.fit(X_Train, Y_Train)

print(logit.cost(X_Test, Y_Test, w))
print (w)
y_Pred = logit.sigmoid(X_Test @ w)
y_Pred = np.where(y_Pred >= .5, 1, -1)

correctPred=0
for i in range (y_Pred.shape[0]) :
    if (Y_Test[i] == y_Pred[i]):
        correctPred += 1

accuracy = correctPred/Y_Test.shape[0] * 100
print(accuracy, "%")