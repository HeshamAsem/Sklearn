#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#----------------------------------------------------

#load boston data

BostonData = load_boston()

#X Data
X = BostonData.data

#y Data
y = BostonData.target

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)


#----------------------------------------------------
#Applying Pipeline :  

#=======================================================================
#Example : 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

steps = [
        ('scalar', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LinearRegression())
        ]
#=======================================================================
PipelineModel = Pipeline(steps)
PipelineModel.fit(X_train, y_train)

#Calculating Details
print('Pipeline Model Train Score is : ' , PipelineModel.score(X_train, y_train))
print('Pipeline Model Test Score is : ' , PipelineModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = PipelineModel.predict(X_test)
print('Predicted Value for Pipeline Model is : ' , y_pred[:10])
