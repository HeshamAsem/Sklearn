
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

kmean = KMeans(n_clusters= 3 )

kmean.fit(X)

result = kmean.labels_

print(silhouette_score(X , result))


score = []
for n in range(2,11):
    kmean = KMeans(n_clusters= n )
    kmean.fit(X)
    result = kmean.labels_
    print(n , '    '  , silhouette_score(X , result))
    score.append(silhouette_score(X , result))
    
plt.plot(range(2,11) , score)
plt.show()

kmean = KMeans(n_clusters= 4 )

    
y_kmeans = kmean.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'r')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'b')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'g')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'c')

plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s = 100, c = 'y')
plt.show()


