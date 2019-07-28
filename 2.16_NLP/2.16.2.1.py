from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

data = pd.read_csv('mall.csv')
data.head()


df = pd.DataFrame(data)


print('Original dataframe is : \n' ,df )

ohe  = OneHotEncoder()
col = np.array(df['Genre'])
col = col.reshape(len(col), 1)

ohe.fit(col)

newmatrix = ohe.transform(col).toarray()
newmatrix = newmatrix.T

df['Female'] = newmatrix[0]
df['male'] = newmatrix[1]

print('Updates dataframe is : \n' ,df )

