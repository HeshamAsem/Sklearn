import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

centers = [[5,2],[-1,2],[0,6] , [5,5]]

x , points = make_blobs(n_samples=750 , centers=centers , cluster_std=0.4, random_state = 0 )


x = StandardScaler().fit_transform(x)


db = DBSCAN(eps=0.3 , min_samples = 10 )
db.fit(x)

samples = np.zeros_like(db.labels_ , dtype = bool)
samples[db.core_sample_indices_] = True
labels = db.labels_

clusters = len(set(labels)) - (1 if -1 in labels else 0)

print( 'number of clusters  = ',  clusters)
print('Homogeniece  = ' , metrics.homogeneity_score(points , labels ))
print('complteness is ' , metrics.completeness_score(points , labels))
print('v measure = ' , metrics.v_measure_score(points , labels))




plt.scatter(x[:,0] , x[:,1])

plt.show()

