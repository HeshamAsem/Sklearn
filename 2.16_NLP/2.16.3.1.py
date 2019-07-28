    

"""

استخدام sklearn  للتعامل مع النصوص

و المثال هنا 
علي ان هناك عدد من الرسائل عدد منها ايجابي و عدد من سلبي , ونريد من sklearn 
 ان تتدرب عليها لمعرفة العناصر التي تجعل هناك ايجابي من سلبي , وبالتالي حينما
 نمرر لها رسالة جديدة تقوم بتحديدها و معرفة هل هي ايجابية ام سلبية


استدعاء الدالة الخاصة بها , ولو تم تحديد كلمات
 كـ stopwords  فسيقوم بحذفها باعتبارها كلمات زائدة , ولو تم عمل lowercase  علي 
انها فولس فلن يقوم بجعل كل الكلمات small  و سيكون case sensetive 
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
#vect = CountVectorizer(stop_words=['call','you'])
#vect = CountVectorizer(lowercase= False)


simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)

"""
استخراج الكلمات الهامة , بعد عملها كلها سمول , ولا تكرار , بترتيب ابجدي ,
 بعد حذف كلمات stop  المعروفة سلفا (مثل a) , وكذلك حذف علامات الترقيم
"""

# examine the fitted vocabulary
words = vect.get_feature_names()

'''
يقوم الـ sklearn  بتحديد اي كلمات ظهرت في اي جمل , عبر عمل ما يسمي 
مصفوفة البارص , والتي تحدد موقع كل كلمة , و كذلك عدد مرات تكرارها
'''

# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
print(simple_train_dtm)

'''
و يمكن تحويلها لمصفوفة عادية لنفهمها  : 
'''


# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()

'''
او اظهارها في داتافريم
'''

# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())


'''
وأهمية مصفوفة البارص , انه في حالة تواجد 10 الاف كلمة
 مختلفة مثلا في 1000 جملة , فستكون المصفوفة فيها 10 مليون قيمة , وهو ما يجعلها
 كبيرة و بطيئة , لكن فعليا 99% من قيم المصفوفة هي اصفار , لان في كل جملة (صف) 
, يتم فقط اختيار الكلمات الموجودة فيها  , وهي مهما كانت معدودة 

بينما مصفوفة البارص قيمها بسيطة , و تكون اسرع و اقل في الذاكرة 


و هنا نقوم باختبار رسالة جديدة . . 
'''

# example text for model testing
simple_test = ["please don't call me"]

'''
نفس امر transform  يحولها الي مصفوفة بارص و منها جدول عادي
'''

# transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()

# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())

'''
و هنا نلحظ
 ان كلمة don’t  غير موجودة , وهذا لأن الـ sklearn  لم يرها علي الاطلاق في التدريب 
, فهو تعامل معها علي انها كلمة جديدة , ولم يتدرب عليها فقام بحذفها . . 

و الخطوة التالية 
ان نقوم باعطاء البيانات السابقة للتدريب مع قيم 1 او 0 لخوارزم مصنف svm  مثلا ,
 وهنا نقوم بعمل predict  للجملة الحالية لمعرفة هل هي ايجابية ام سلبية
'''