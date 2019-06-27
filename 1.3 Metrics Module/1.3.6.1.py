#Import Libraries
from sklearn.metrics import f1_score
#----------------------------------------------------

#Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)
# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

F1Score = f1_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('F1 Score is : ', F1Score)