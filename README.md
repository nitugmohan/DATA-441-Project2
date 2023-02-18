# Project2

Nitu Girish Mohan

## Gramfort's Approach - taken from class notebook

The idea is based on the following references:

- William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.

- William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.

The main idea is to perform a local linear regression in a neighborhood of k observations close by and then apply an iteration to rescale the weights based on the residuals; this step is argued to yield a more robust model.

Below is Gramfort's Approach:

```python
# approach by Alex Gramfort

# https://gist.github.com/agramfort/850437

"""William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.

William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610."""

def lowess_ag(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest
 ```

## Adapt and modify the code for Gramfort’s version of Lowess to accommodate train and test sets with multidimensional features.

This version of Gramfort's approach accounts for multidimensional features:

```python
def lw_ag_md(x, y, xnew,f=2/3,iter=3, intercept=True): #train data, new data

  n = len(x)
  r = int(ceil(f * n))
  yest = np.zeros(n)

  if len(y.shape)==1: # here we make column vectors
    y = y.reshape(-1,1)

  if len(x.shape)==1:
    x = x.reshape(-1,1)
  
  if intercept:
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else:
    x1 = x

  # We compute the max bounds for the local neighborhoods
  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]

  w = np.clip(dist(x,x) / h, 0.0, 1.0)
  # Opportunity for more research --> change the kernel
  w = (1 - w ** 3) ** 3

  #Looping through all X-points
  delta = np.ones(n)
  for iteration in range(iter):
    for i in range(n):
      W = np.diag(w[:,i])
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      ##
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization
      beta = linalg.solve(A, b)
      #beta, res, rnk, s = linalg.lstsq(A, b)
      yest[i] = np.dot(x1[i],beta)

    residuals = y - yest
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    delta = (1 - delta ** 2) ** 2

  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      # the trick below is to have the Delauney triangulation work
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,y[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,y.ravel()) 
    # output[np.isnan(output)] = g(X[np.isnan(output)])
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
  ```
  
  ```python
  xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
  yhat = lw_ag_md(xtrain,ytrain,xtest,f=1/50,iter=3,intercept=True)
  mse(ytest,yhat)
  ```






## Test your new function on some real data sets with k-Fold cross-validations

```python
#Import statements + functions
# graphical libraries
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
from IPython.display import Image
from IPython.display import display
plt.style.use('seaborn-white')

!pip install --upgrade --q scipy

# computational libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import scipy.stats as stats 
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from math import ceil
from scipy import linalg
# the following line(s) are necessary if you want to make SKlearn compliant functions
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

lm = LinearRegression()
scale = StandardScaler()
qscale = QuantileTransformer()

# here we have a function that computes the Euclidean distance between all the observations in u, and v
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
  return d
```  

### With the Cars Data:

```python
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/22 23 - Junior Yr/Adv Applied Machine Learning/Preliminaries, Intro to Locally Weighted Regression/cars.csv')
x = data.loc[:,'CYL':'WGT'].values
y = data['MPG'].values
```

```python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
model_lw = Lowess_AG_MD(f=1/3,iter=1,intercept=True)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model_lw.fit(xtrain,ytrain)
  yhat_lw = model_lw.predict(xtest)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

The Cross-validated Mean Squared Error for Locally Weighted Regression is : 23.53685564215548'

The Cross-validated Mean Squared Error for Random Forest is : 16.95579463156351

### With the Concrete Data:

```python
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/22 23 - Junior Yr/Adv Applied Machine Learning/Preliminaries, Intro to Locally Weighted Regression/concrete.csv')
x = data.loc[:,'cement':'age'].values
y = data['strength'].values
``` 

```python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
model_lw = Lowess_AG_MD(f=1/3,iter=1,intercept=True)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model_lw.fit(xtrain,ytrain)
  yhat_lw = model_lw.predict(xtest)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```
The Cross-validated Mean Squared Error for Locally Weighted Regression is : 66.91415418086902

The Cross-validated Mean Squared Error for Random Forest is : 45.09761506471062

## Create a SciKitLearn-compliant version of the function you wrote for 1) and test it with GridSearchCV from SciKitLearn. 
```python
class Lowess_AG_MD:
    def __init__(self, f = 1/10, iter = 3,intercept=True):
        self.f = f
        self.iter = iter
        self.intercept = intercept
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        return lw_ag_md(x, y, x_new, f, iter, intercept)

    def get_params(self, deep=True):
    # suppose this estimator has parameters "f", "iter" and "intercept"
        return {"f": self.f, "iter": self.iter,"intercept":self.intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
        
model = Lowess_AG_MD(f=1/5,iter=3,intercept=True)
model.fit(xtrain,ytrain)
yhat = model.predict(xtest)
mse(ytest,yhat)
```

```python
lwr_pipe = Pipeline([('zscores', StandardScaler()),
                     ('lwr', Lowess_AG_MD())])
params = [{'lwr__f': [1/i for i in range(3,15)],
         'lwr__iter': [1,2,3,4]}]
gs_lowess = GridSearchCV(lwr_pipe,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5)
gs_lowess.fit(x, y)
gs_lowess.best_params_
gs_lowess.score(x,y)
```
-1.6547449275510306


I referenced the class notebooks for this project.
