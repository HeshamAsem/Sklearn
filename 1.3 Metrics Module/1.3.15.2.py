from sklearn.metrics import zero_one_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]

zero_one_loss(y_true, y_pred)


zero_one_loss(y_true, y_pred, normalize=False)
