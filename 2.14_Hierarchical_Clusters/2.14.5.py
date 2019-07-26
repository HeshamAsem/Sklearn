
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# generating two clusters: x with 10 points and y with 20:
#np.random.seed(1234)
x = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[10,])
y = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[20,])
x
y

X = np.concatenate((x, y),)
X
print(X.shape)  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])
plt.show()

 
# generate the linkage matrix
Z = linkage(X, 'ward')
#print(Z)


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
coph_dists = cophenet(Z, pdist(X))
#coph_dists


plt.figure(figsize=(10, 5))
plt.title('HCA Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z,leaf_rotation=90,leaf_font_size=12,)
plt.show()


