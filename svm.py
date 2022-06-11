#!pip install cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers
import numpy as np

class SVM:
  def __init__(self):
    self.w = None
    self.b = None
  def train(self,X,y):
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1

    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    solution = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solution['x'])
    w = ((alpha * y).T @X).reshape(-1, 1)
    seuil = 0.0
    s = (alpha > seuil).flatten()
    b = y[s] - np.dot(X[s], w)
    self.w = w
    self.b =b
    return w, b 
  def test(self,X):
    print (X.shape)
    n = X.shape[0]
    y_pred = np.zeros((n,1))

    for i in range(n):
      distance=np.dot(np.array(X[i]), self.w) + self.b
    #print (distance.shape)
      y_pred[i] = np.sign(distance)[0]
    return y_pred

  def accuracy(self,y_pred,y):
      n= y_pred.shape[0]
      good_pred = 0.0
      for i in range(n):
        if y_pred[i] == y[i]:
          good_pred +=1
      return good_pred/n    