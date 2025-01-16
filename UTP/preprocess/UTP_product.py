import numpy as np
import pandas as pd
import pickle
import random
import xgboost as xgb
import os
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score
warnings.filterwarnings("ignore")

data_train = pd.read_pickle("data/df_train.pkl")
data_test = pd.read_pickle("data/df_test.pkl")

X_train = np.array(data_train["ESM1b"].tolist())
Y_train = np.array(data_train["label"])
X_test = np.array(data_test["ESM1b"].tolist())
Y_test = np.array(data_test["label"])
features = ["ESM1b_" + str(i) for i in range(1280)]

all_clusters_train = list(set(data_train["cluster"]))
random.shuffle(all_clusters_train)
n = len(all_clusters_train)
k = int(n / 5)
clusters_fold1 = all_clusters_train[:k]
clusters_fold2 = all_clusters_train[k:k * 2]
clusters_fold3 = all_clusters_train[k * 2:k * 3]
clusters_fold4 = all_clusters_train[k * 3:k * 4
clusters_fold5 = all_clusters_train[k * 4:]
fold_indices = [list(data_train.loc[data_train["cluster"].isin(clusters_fold1)].index),
                list(data_train.loc[data_train["cluster"].isin(clusters_fold2)].index),
                list(data_train.loc[data_train["cluster"].isin(clusters_fold3)].index),
                list(data_train.loc[data_train["cluster"].isin(clusters_fold4)].index),
                list(data_train.loc[data_train["cluster"].isin(clusters_fold5)].index)]

train_indices = [[], [], [], [], []]
test_indices = [[], [], [], [], []]

for i in range(5):
    for j in range(5):
        if i != j:
            train_indices[i] = train_indices[i] + fold_indices[j]
    test_indices[i] = fold_indices[i]
  
param = {
    'learning_rate': 0.3400719327428487,
    'max_delta_step': 0.4515553136138156,
    'max_depth': 2,
    'min_child_weight': 2.271002905005067,
    'num_rounds': 372.4732061978861,
    'reg_alpha': 1.2700148259733535,
    'reg_lambda': 0.6890954213241562,
    'weight': 0.36279298717024
    'tree_method':'gpu_hist', 
    'sampling_method':'gradient_based',
    'objective':'binary:logistic',
         }
         
num_round = param["num_rounds"]
weights = np.array([param["weight"] if outcome == 0 else 1.0 for outcome in data_train["outcome"]])
del param["num_rounds"]
del param["weight"]

mcc = []
roc_auc = []
acc = []
for i in range(5):
    train_index, test_index = train_indices[i], test_indices[i]
    data_train = xgb.DMatrix(np.array(X_train[train_index]), label=np.array(Y_train[train_index]))
    dvalid = xgb.DMatrix(np.array(X_train[test_index]))
    bst = xgb.train(param, data_train, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = Y_train[test_index]
    acc.append(np.mean(y_valid_pred == np.array(validation_y)))
    roc_auc.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    mcc.append(matthews_corrcoef(np.array(validation_y), y_valid_pred))

print(f"Mean Accuracy: {np.mean(acc)}")
print(f"Mean ROC-AUC: {np.mean(roc_auc)}")
print(f"Mean MCC: {np.mean(mcc)}")

data_trainset = xgb.DMatrix(np.array(X_train), weight=weights, label=np.array(Y_train), feature_names=features)
data_testset = xgb.DMatrix(np.array(X_test), label=np.array(Y_test), feature_names=features)
bst = xgb.train(param, data_trainset, int(num_round))

y_test_pred = np.round(bst.predict(data_testset))
accuracy = np.mean(y_test_pred == np.array(Y_test))
roc_auc = roc_auc_score(np.array(Y_test), bst.predict(data_testset))
mcc = matthews_corrcoef(np.array(Y_test), y_test_pred)
recall = recall_score(np.array(Y_test), y_test_pred)
precision = precision_score(np.array(Y_test), y_test_pred)

data_test['UTP_Pred_label'] = y_test_pred 
data_test.to_csv('../result/UTP_result_Testset.csv', index=False)

print(f"Final Accuracy: {accuracy}")
print(f'Final Precision: {precision}')
print(f'Final Recall: {recall}')
print(f"Final ROC-AUC: {roc_auc}")
print(f"Final MCC: {mcc}")

model_file = 'model/UTP.dat'
with open(model_file, 'wb') as f:
    pickle.dump(bst, f)