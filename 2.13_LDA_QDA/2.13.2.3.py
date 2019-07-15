
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = np.random.randint(20, size=(50, 10))
y = np.random.randint(5, size=(50, 1))

Qclf = QuadraticDiscriminantAnalysis()
Qclf.fit(X, y)

QDAScore = Qclf.score(X,y)
print('Quadratic Score = ' , QDAScore )

z = np.random.randint(20, size=(1, 10))
print('Quadratic Prediction = ' , Qclf.predict(z))

###########################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


Lclf = LinearDiscriminantAnalysis()
Lclf.fit(X, y)

Lclf.score(X,y)

LDAScore = Lclf.score(X,y)
print('Linear Score = ' , LDAScore )

z = np.random.randint(20, size=(1, 10))
print('Linear Prediction = ' , Lclf.predict(z))

