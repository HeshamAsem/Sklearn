from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('mall.csv')
data.head()


df = pd.DataFrame(data)


print('Original dataframe is : \n' ,df )


enc  = LabelEncoder()
enc.fit(df['Genre'])


print('classed found : ' , list(enc.classes_))

print('equivilant numbers are : ' ,enc.transform(df['Genre']) )

df['Genre Code'] = enc.transform(df['Genre'])

print('Updates dataframe is : \n' ,df )

print('Inverse Transform  : ' ,list(enc.inverse_transform([1,0,1,1,0,0])))
