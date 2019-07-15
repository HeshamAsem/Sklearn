import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

X
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

classifier.score(X_test , y_test)

y_pred = classifier.predict(np.array([48,0,2,.130,0.275,0,1,1.39,0,0.2,2,0,2]).reshape(1,-1))
y_pred 

y_pred = classifier.predict(X_test)
y_pred 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()

  
