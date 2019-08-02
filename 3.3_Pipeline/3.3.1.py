#Import Libraries
from sklearn.pipeline import Pipeline
#----------------------------------------------------

#Applying Pipeline :  

#=======================================================================
#Example : 

#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import StandardScaler

#steps = [
#        ('scalar', StandardScaler()),
#        ('poly', PolynomialFeatures(degree=2)),
#        ('model', LinearRegression())
#        ]
#=======================================================================
PipelineModel = Pipeline(steps)
PipelineModel.fit(X_train, y_train)

#Calculating Details
print('Pipeline Model Train Score is : ' , PipelineModel.score(X_train, y_train))
print('Pipeline Model Test Score is : ' , PipelineModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = PipelineModel.predict(X_test)
y_pred_proba = PipelineModel.predict_proba(X_test)
print('Predicted Value for Pipeline Model is : ' , y_pred[:10])
print('Predicted Probabilities Values for Pipeline Model is : ' , y_pred_proba[:10])