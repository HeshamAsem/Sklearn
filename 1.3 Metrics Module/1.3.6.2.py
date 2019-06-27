from sklearn.metrics import f1_score
y_pred = [0, 2, 1, 0, 0, 1]
y_true = [0, 1, 2, 0, 1, 2]
f1_score(y_true, y_pred, average='micro')
