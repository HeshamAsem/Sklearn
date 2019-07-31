from sklearn import datasets

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict


diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]



model1 = LinearRegression()
model2 = SVR(gamma = 'auto')
model3 = DecisionTreeRegressor()
model4 = RandomForestRegressor(n_estimators = 20)



models = [model1 , model2 , model3 , model4]

x=0
for m in models:
    x+=1
    
    for n in range(2,5):
        print('result of model number : ' , x ,' for cv value ',n,' is \n' , cross_val_predict(m, X, y, cv=n))  
        print('-----------------------------------')
    print('=====================================')
    print('=====================================')


