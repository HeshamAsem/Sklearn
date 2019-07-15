import pandas as pd

df = pd.read_csv('diabetes.csv')

df.head()

df.shape



X = df.drop(columns=['Outcome'])
X.head()


y = df['Outcome'].values
y[0:5]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)


knn.predict(X_test)[0:5]


knn.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
import numpy as np
knn_cv = KNeighborsClassifier(n_neighbors=3)

cv_scores = cross_val_score(knn_cv, X, y, cv=5)

print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
 