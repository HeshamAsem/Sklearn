#Import Libraries
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#----------------------------------------------------

#Applying VotingClassifier Model 

'''
ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)
'''

#loading models for Voting Classifier
RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=1, random_state=33)
LDAModel_ = LinearDiscriminantAnalysis(n_components=3 ,solver='svd')

#loading Voting Classifier
VotingClassifierModel = VotingClassifier(estimators=[('RFModel',RFModel_),('LDAModel',LDAModel_)], voting='hard')
VotingClassifierModel.fit(X_train, y_train)

#Calculating Details
print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = VotingClassifierModel.predict(X_test)
print('Predicted Value for VotingClassifierModel is : ' , y_pred[:10])