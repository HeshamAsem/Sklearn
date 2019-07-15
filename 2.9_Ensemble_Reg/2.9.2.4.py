import pandas as pd
features = pd.read_csv('data.csv')
features.head(5)

print('The shape of our features is:', features.shape)

features.describe()

features = pd.get_dummies(features)
features.iloc[:,5:].head(5)

import numpy as np
labels = np.array(features['actual'])

features= features.drop('actual', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

from sklearn.ensemble import  GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators = 100 , learning_rate = 1.5 , max_depth = 1)
model.fit(train_features, train_labels)


predictions = model.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



