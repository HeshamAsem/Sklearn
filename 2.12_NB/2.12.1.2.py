#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------

#load breast cancer data

BreastData = load_breast_cancer()

#X Data
X = BreastData.data
#print('X Data is \n' , X[:10])
#print('X shape is ' , X.shape)
#print('X Features are \n' , BreastData.feature_names)

#y Data
y = BreastData.target
#print('y Data is \n' , y[:10])
#print('y shape is ' , y.shape)
#print('y Columns are \n' , BreastData.target_names)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying GaussianNB Model 

'''
#sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
'''

GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)

#Calculating Details
print('GaussianNBModel Train Score is : ' , GaussianNBModel.score(X_train, y_train))
print('GaussianNBModel Test Score is : ' , GaussianNBModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)
print('Predicted Value for GaussianNBModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for GaussianNBModel is : ' , y_pred_prob[:10])

#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()