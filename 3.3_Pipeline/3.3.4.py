
from sklearn import datasets
import matplotlib.pyploy as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

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

 
pipe.fit(X_tr, y_tr)
print("Train Score (CV score=%0.3f):" % pipe.score(X_tr, y_tr))
print("Test Score (CV score=%0.3f):" % pipe.score(X_ts, y_ts))
 
y_pred = pipe.predict(X_ts)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_ts, y_pred)

import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()
