#Import Libraries
from sklearn.metrics import roc_auc_score
#----------------------------------------------------

#Calculating ROC AUC Score:  
#roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)

ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
#print('ROCAUC Score : ', ROCAUCScore)