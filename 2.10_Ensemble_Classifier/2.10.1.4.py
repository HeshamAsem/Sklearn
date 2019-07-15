import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()

#####################################################################


for j in range(2,100):
   
    classifier = RandomForestClassifier(n_estimators = j, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print('RF for number of Trees : ' , j , ' is : \n' , cm)
    print('The Score is : ',classifier.score(X_test , y_test))
    print('=======================================================')