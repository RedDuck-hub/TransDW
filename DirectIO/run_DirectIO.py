import numpy as np
import pandas as pd
import pickle
import esm
import torch
import random
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

try:
    model_file = 'preprocess/model/DirectIO.dat'
    with open(model_file, "rb") as f:
        rf = pickle.load(f)
    
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

    transporters = pd.read_csv("prediction/Testset3.csv")
    transporters = convert_vectors(transporters)
    
    X_test = np.array(transporters["ESM1b"].tolist())
    probabilities = rf.predict_proba(X_test)
    probabilities_rounded = np.round(probabilities, 4)
    print(probabilities_rounded)
    probabilities_df = pd.DataFrame(probabilities_rounded, columns=['Probability_Class_0', 'Probability_Class_1'])
    y_test_pred = probabilities_df['Probability_Class_1']
    transporters['DirectIO_Result'] = y_test_pred
    
    transporters['Pred_label'] = ''
    for ind in transporters.index:
        if transporters['result'][ind] >= 0.5:
            transporters.loc[ind, 'Pred_label'] = 1
        else:
            transporters.loc[ind, 'Pred_label'] = 0
    		
    Y_test = transporters['Label'].to_list()
    Y_Pred = transporters['Pred_label'].to_list()
    Y_Pred_Prob = transporters['DirectIO_Result'].to_list()
    accuracy = accuracy_score(Y_test, Y_Pred)
    roc_auc = roc_auc_score(Y_test, Y_Pred_Prob)
    mcc = matthews_corrcoef(Y_test, Y_Pred)
    ba = balanced_accuracy_score(Y_test, Y_Pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"MCC: {mcc}")
    print(f"Balanced Accuracy: {ba}")
    transporters.to_csv("result/DirectIO_result_Testset3.csv", index=False)


except Exception as e:
    print(f"An error occurred: {e}")
