import pandas as pd
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('data.csv')

neigh = NearestNeighbors(2, 0.4)
neigh.fit(data)

data
l =[-2,.5,-0.8,1.1,1.5,0.1,-1,2]

result = neigh.kneighbors([l],n_neighbors= 2) # returns distance nd index
 