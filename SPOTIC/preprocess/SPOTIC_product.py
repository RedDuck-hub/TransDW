import numpy as np
import pandas as pd
import pickle
import random
import os
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score
warnings.filterwarnings("ignore")

data_train = pd.read_pickle("data/df_final_train_with_clusters_with_ESM1b.pkl")
data_test = pd.read_pickle("data/df_final_test_with_ESM1b.pkl")
data_train = data_train.reset_index()
data_test = data_test.reset_index()

all_clusters_train = list(set(data_train["cluster"]))
random.shuffle(all_clusters_train)
n = len(all_clusters_train)
k = int(n / 5)
clusters_fold1 = all_clusters_train[:k]
clusters_fold2 = all_clusters_train[k:k * 2]
clusters_fold3 = all_clusters_train[k * 2:k * 3]
clusters_fold4 = all_clusters_train[k * 3:k * 4]
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
 
train_X = []
train_y = []
test_X = []
test_y = []
for ind in data_train.index:
    emb = data_train["ESM1b"][ind]
    ecfp = np.array(list(data_train["ECFP"][ind])).astype(int)
    train_X.append(np.concatenate([ecfp, emb]))
    train_y.append(int(data_train["outcome"][ind]))
    
for ind in data_test.index:
    emb = data_test["ESM1b"][ind]
    ecfp = np.array(list(data_test["ECFP"][ind])).astype(int)
    test_X.append(np.concatenate([ecfp, emb]))
    test_y.append(int(data_test["outcome"][ind]))

train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)
feature_names = ["ECFP_" + str(i) for i in range(1024)]
feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

param = {'learning_rate': 0.25712750071519765,
         'max_delta_step': 3.3218724218670506,
         'max_depth': 12,
         'min_child_weight': 0.28866353603424871,
         'num_rounds': 157.0835284747203,
         'reg_alpha': 0.7421715909450698,
         'reg_lambda': 0.2391747120404815,
         'weight': 0.31857913264004466,
         'tree_method':'gpu_hist', 
         'sampling_method':'gradient_based',
         'objective':'binary:logistic',
         }

num_round = param["num_rounds"]
weights = np.array([param["weight"] if outcome == 0 else 1.0 for outcome in data_train["outcome"]])
del param["num_rounds"]
del param["weight"]

mcc = []
acc = []
roc_auc = []
for i in range(5):
    train_index, test_index = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), label=np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]
    acc.append(np.mean(y_valid_pred == np.array(validation_y)))
    roc_auc.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    mcc.append(matthews_corrcoef(np.array(validation_y), y_valid_pred))

print(f"Mean Accuracy: {np.mean(acc)}")
print(f"Mean ROC-AUC: {np.mean(roc_auc)}")
print(f"Mean MCC: {np.mean(mcc)}")

dtrain = xgb.DMatrix(np.array(train_X), weight=weights, label=np.array(train_y), feature_names=feature_names)
dtest = xgb.DMatrix(np.array(test_X), label=np.array(test_y), feature_names=feature_names)
bst = xgb.train(param, dtrain, int(num_round))

y_test_pred = np.round(bst.predict(dtest))
accuracy = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)
recall = recall_score(np.array(test_y), y_test_pred)
precision = precision_score(np.array(test_y), y_test_pred)

print(f"Final Accuracy: {accuracy}")
print(f'Final Precision: {precision}')
print(f'Final Recall: {recall}')
print(f"Final ROC-AUC: {roc_auc}")
print(f"Final MCC: {mcc}")

data_test['SPOTIC_Pred_label'] = y_test_pred
del data_test['ESM1b'] 
del data_test['ECFP']
data_test.to_csv('../result/spotic_Testset_result.csv', index=False)
model_file = 'model/SPOTIC.dat'
with open(model_file, 'wb') as f:
    pickle.dump(bst, f)
