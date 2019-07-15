#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
#Applying GradientBoostingClassifier Model 

'''
ensemble.GradientBoostingClassifier(loss='deviance’, learning_rate=0.1,n_estimators=100, subsample=1.0,
                                    criterion='friedman_mse’,min_samples_split=2,min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0,max_depth=3,min_impurity_decrease=0.0,
                                    min_impurity_split=None,init=None, random_state=None,max_features=None,
                                    verbose=0, max_leaf_nodes=None,warm_start=False, presort='auto’, 
                                    validation_fraction=0.1,n_iter_no_change=None, tol=0.0001)
'''

GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 
GBCModel.fit(X_train, y_train)

#Calculating Details
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
print('GBCModel features importances are : ' , GBCModel.feature_importances_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = GBCModel.predict(X_test)
y_pred_prob = GBCModel.predict_proba(X_test)
print('Predicted Value for GBCModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])

#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()