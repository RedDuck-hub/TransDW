import numpy as np
import pandas as pd
import pickle
import random
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

data_train = pd.read_pickle("data/df_train.pkl")
data_test = pd.read_pickle("data/df_test.pkl")
data_train = data_train[['label', 'ESM1b', 'cluster']]
data_test = data_test[['ID', 'Uniprot ID', 'Sequence', 'label', 'ESM1b']]

X_train = np.array(data_train["ESM1b"].tolist())
Y_train = np.array(data_train["label"])
X_test = np.array(data_test["ESM1b"].tolist())
Y_test = np.array(data_test["label"])

all_clusters_train = list(set(data_train["cluster"]))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = []
for train_index, test_index in kf.split(X_train, data_train["cluster"]):
    fold_indices.append((train_index, test_index))
    
rf_param = {
    'n_estimators': 127,  
    'max_depth': None, 
    'min_samples_split': 3,
    'min_samples_leaf': 3, 
    'random_state': 42, 
    'class_weight': 'balanced'
}
rf = RandomForestClassifier(**rf_param)

acc = []
roc_auc = []
mcc = []
datas = pd.DataFrame(X_train)
for train_index, test_index in fold_indices:
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = Y_train[train_index], Y_train[test_index]
    rf.fit(X_train_fold, y_train_fold) 
    y_pred = rf.predict(X_test_fold)
    
    acc.append(np.mean(y_pred == y_test_fold))
    roc_auc.append(roc_auc_score(y_test_fold, rf.predict_proba(X_test_fold)[:, 1]))
    mcc.append(matthews_corrcoef(y_test_fold, y_pred))

print(f"Mean Accuracy: {np.mean(acc)}")
print(f"Mean ROC-AUC score: {np.mean(roc_auc)}")
print(f"Mean MCC: {np.mean(mcc)}")

rf.fit(X_train, Y_train)
y_test_pred = rf.predict(X_test)
accuracy = np.mean(y_test_pred == Y_test)
roc_auc = roc_auc_score(Y_test, rf.predict_proba(X_test)[:, 1])
mcc = matthews_corrcoef(Y_test, y_test_pred)
ba = balanced_accuracy_score(Y_test, y_test_pred)

data_test['DirectIO_Result'] = y_test_pred
del data_test['ESM1b']
data_test.to_csv('../result/DirectIO_result_Testset.csv',index=False)

print(f"Final Accuracy: {accuracy}")
print(f"Final Balanced Accuracy: {ba}")
print(f"Final ROC-AUC: {roc_auc}")
print(f"Final MCC: {mcc}")

model_file = 'model/DirectIO.dat'
with open(model_file, 'wb') as f:
    pickle.dump(rf, f)