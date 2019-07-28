

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# read file into pandas using a relative path

sms = pd.read_table('sms.tsv', header=None, names=['label', 'message'])

# examine the shape
sms.shape

# examine the first 10 rows
sms.head(10)

# examine the class distribution
sms.label.value_counts()

'''
عمل عمود جديد له باسم label_num  يكون فيه 0 او 1 حسب حقيقي ام لا
'''

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
sms.shape

# check that the conversion worked
sms.head(10)


# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# instantiate the vectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix


vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

'''
عمل مصفوفة بارص
'''

# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)

'''
مشاهدة المصفوفة الحقيقية
'''

# examine the document-term matrix
X_train_dtm.toarray()

pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())

'''
التجريب علي بيانات الاختبار
'''

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm.toarray()
pd.DataFrame(X_test_dtm.toarray(), columns=vect.get_feature_names())

'''
استخدام نايف بايز الملتينوميال
'''

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# train the model using X_train_dtm (timing it with an IPython "magic command")
nb.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# example false negative
X_test[3132]



# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)
 

