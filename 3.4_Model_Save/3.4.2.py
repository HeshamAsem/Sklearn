
import sklearn.svm as s
import pickle as pk
import numpy as np


x = np.random.randint(10,size =20).reshape(4,5)
y = [5,8,9,6]


model = s.SVR()
model.fit(x,y)


pk.dump(model , open('saved file2.sav','wb'))

model.predict([[2,3,6,5,9]])
##############################################

savedmodel = pk.load(open('saved file2.sav','rb'))
savedmodel.predict([[2,3,6,5,9]])
