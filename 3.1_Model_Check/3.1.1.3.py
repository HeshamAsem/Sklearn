from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
reg= linear_model.LinearRegression()

cv_results = cross_validate(reg, X, y, cv=3,return_train_score=False)


for key in cv_results.keys():
    print('value of ' , key , ' is  ' , cv_results[key])

scores = cross_validate(reg, X, y, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)

print('details are  : \n' , scores)
