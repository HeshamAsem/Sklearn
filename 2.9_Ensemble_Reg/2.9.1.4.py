
import numpy as np
import pandas as pd

dataset = pd.read_csv('houses.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X)
X= imp.transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)




