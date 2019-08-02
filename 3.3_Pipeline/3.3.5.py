
import numpy as np
#import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
logistic = SGDClassifier(loss='log', penalty='l2', 
                         max_iter=10000, tol=1e-5, random_state=0)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
digits = datasets.load_digits()


X_tr = digits.data[:1200,:]
y_tr = digits.target[:1200]
X_ts = digits.data[1200:,:]
y_ts = digits.target[1200:]

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {'pca__n_components': [5, 20, 30, 40, 50, 64],
              'logistic__alpha': np.logspace(-4, 4, 5),}
              

search = GridSearchCV(pipe, param_grid, iid=False, cv=5,return_train_score=False)

search.fit(X_tr, y_tr)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

 
