
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

y_pred1 = classifier1.predict(X_test)

score1 = classifier1.score(X_test, y_test)

print('score 1 = ', score1 )


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)

print('cm1 \n' , cm1 )

import seaborn as sns
sns.heatmap(cm1, center=True)
plt.show()

#####################################################################
 # apply LDA 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)

score2 = classifier2.score(X_test, y_test)

print('score 2 = ', score2 )


cm2 = confusion_matrix(y_test, y_pred2)
print('cm2 \n' , cm2 )

sns.heatmap(cm2, center=True)
plt.show()


