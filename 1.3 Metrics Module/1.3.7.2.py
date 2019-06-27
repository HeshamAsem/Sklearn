from sklearn.metrics import recall_score
y_pred =  ['a','b','c','a','b','c','a','b','c','a']
y_true =   ['a','a','b','b','a','b','c','c','b','b']
recall_score(y_true, y_pred, average=None)
