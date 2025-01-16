import numpy as np
import pandas as pd
import pickle
import random
import os
import xgboost as xgb
import esm
import torch
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

try:
    model_path = "preprocess/model/SPOTIC.dat"
    with open(model_file, "rb") as f:
        bst = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.to(device)
    batch_size=10
    batch_converter = alphabet.get_batch_converter()

    def convert_ESM1b(df_transporter):
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

    df_transporter = pd.read_csv("prediction/Testset4.csv")
    df_transporter = convert_ESM1b(df_transporter)

    df_transporter["ECFP"] = ""
    for ind in df_transporter.index:
        mol = Chem.inchi.MolFromInchi(df_transporter["InChI"][ind])
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToBitString()
        df_transporter["ECFP"][ind] = ecfp

    test_X = []
    for ind in df_transporter.index:
        emb = df_transporter["ESM1b"][ind]
        ecfp = np.array(list(df_transporter["ECFP"][ind])).astype(int)
        test_X.append(np.concatenate([ecfp, emb]))
    test_X = np.array(test_X)

    feature_names = ["ECFP_" + str(i) for i in range(1024)]
    feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
    dtest = xgb.DMatrix(np.array(test_X), feature_names=feature_names)
    y_test_pred = bst.predict(dtest)

    df_transporter['SPOTIC_Result'] = y_test_pred
    del df_transporter['ESM1b'] 
    del df_transporter['ECFP']
    df_transporter.to_csv("result/spotic_Testset4_result.csv", index=False)

except Exception as e:
    print(f"An error occurred: {e}")