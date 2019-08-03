
import sklearn.svm as s
import sklearn.externals.joblib as jb
import numpy as np


x = np.random.randint(10,size =20).reshape(4,5)
y = [5,8,9,6]


model = s.SVR()
model.fit(x,y)


jb.dump(model , 'saved file.sav')

model.predict([[2,3,6,5,9]])
##############################################

savedmodel = jb.load('saved file.sav')
savedmodel.predict([[2,3,6,5,9]])

