
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rn = np.random.RandomState()
x = np.dot(rn.rand(2,2) , rn.randn(2,100) ).T
x.shape

model = PCA(n_components= 1)
model.fit(x)

data = model.transform(x)

x.shape
data.shape


newdata = model.inverse_transform(data)

plt.scatter(x[:,0],x[:,1])
plt.scatter(newdata[:,0],newdata[:,1])
 
