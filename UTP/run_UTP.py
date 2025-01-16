import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import esm
import torch
import random
import os
import warnings
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score
from rdkit import Chem
from rdkit.Chem import AllChem
warnings.filterwarnings("ignore")

try:
    model_file = 'preprocess/model/UTP.dat'
    with open(model_file, "rb") as f:
        bst = pickle.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.to(device)
    batch_size=5
    batch_converter = alphabet.get_batch_converter()

    def convert_vectors(df_transporter):
        df_transporter["ESM1b"] = ""
        target_seqs = list(set(df_transporter['Sequence']))
        targets = [(str(i), seq) for i, seq in enumerate(target_seqs)]
    
        for i in tqdm(range(0, len(targets), batch_size)):
            batch = targets[i:i+batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)

            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
   
                for j, item in enumerate(batch):
                    result = results["representations"][33][j, 1: len(item[1]) + 1].mean(0).cpu().numpy()
                    target_rows = df_transporter[df_transporter['Sequence'] == item[1]]
                    for index in target_rows.index:
                        df_transporter.at[index, "ESM1b"] = result
        return df_transporter
        
    transporters = pd.read_csv("prediction/Testset2.csv")
    transporters = convert_vectors(transporters)
    
    X_test = []
    for ind in transporters.index:
        emb = transporters["ESM1b"][ind]
        X_test.append(emb)
    
    features = ["ESM1b_" + str(i) for i in range(1280)]
    y_test_pred = bst.predict(xgb.DMatrix(np.array(X_test), feature_names=features))

    transporters['UTP_Result'] = y_test_pred
    transporters['UTP_Pred_label'] = ''
    for ind in transporters.index:
        if transporters['UTP_Result'][ind] >= 0.5:
            transporters.loc[ind, 'UTP_Pred_label'] = 1
        else:
            transporters.loc[ind, 'UTP_Pred_label'] = 0
    		
    Y_test = transporters['Label'].to_list()
    Y_Pred = transporters['UTP_Pred_label'].to_list()
    Y_Pred_Prob = transporters['UTP_Result'].to_list()
    
    accuracy = accuracy_score(Y_test, Y_Pred)
    roc_auc = roc_auc_score(Y_test, Y_Pred_Prob)
    mcc = matthews_corrcoef(Y_test, Y_Pred)
    recall = recall_score(Y_test, Y_Pred)
    precision = precision_score(Y_test, Y_Pred)

    print(f"Accuracy: {accuracy}")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f"ROC-AUC: {roc_auc}")
    print(f"MCC: {mcc}")
    
    transporters.to_csv("result/UTP_result_Testset2.csv", index=False)

except Exception as e:
    print(f"An error occurred: {e}")
