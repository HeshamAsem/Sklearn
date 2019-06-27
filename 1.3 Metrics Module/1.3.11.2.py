from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2,5]
y_pred = [0, 0, 2, 2, 1,0]
print(classification_report(y_true, y_pred ))

#==========================================================


y_true = ['a','d','a','g','a','d']
y_pred = ['a','a','g','g','d','g']
print(classification_report(y_true, y_pred ))
