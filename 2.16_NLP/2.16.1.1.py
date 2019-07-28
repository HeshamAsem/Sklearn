from sklearn import preprocessing
import pandas as pd
raw_data = {'patient': [1, 1, 1, 2, 2],
        'obs': [1, 2, 3, 1, 2],
        'treatment': [0, 1, 0, 1, 0],
        'score': ['strong', 'weak', 'normal', 'weak', 'strong']}
df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])


print('Original dataframe is : \n' ,df )

# Create a label (category) encoder object
le = preprocessing.LabelEncoder()
# Fit the encoder to the pandas column
le.fit(df['score'])


print('classed found : ' , list(le.classes_))

print('equivilant numbers are : ' ,le.transform(df['score']) )

df['score'] = le.transform(df['score'])

print('Updates dataframe is : \n' ,df )

print('Inverse Transform  : ' ,list(le.inverse_transform([2, 2, 1])))
