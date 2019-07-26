import matplotlib.pyplot as plt  
import pandas as pd  



customer_data = pd.read_csv('shopping_data.csv')  


customer_data.shape

customer_data.head()  

data = customer_data.iloc[:, 3:5].values  


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(8, 6))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward')) 


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data) 
