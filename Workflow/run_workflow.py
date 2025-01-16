import numpy as np
import pandas as pd
import pickle
import random
import os
import sys
import time
import subprocess
from io import StringIO
from os.path import join
import xgboost as xgb
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import esm
import re
import string
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# substrate inchi
inchi = 'InChI=1S/C11H15N2O8P/c12-10(16)6-2-1-3-13(4-6)11-9(15)8(14)7(21-11)5-20-22(17,18)19/h1-4,7-9,11,14-15H,5H2,(H3-,12,16,17,18,19)/t7-,8-,9-,11-/m1/s1'
# The minimum threshold of sequence similarity
identity = int(20)
# Input file containing the target transporters to be predicted
file = 'templates/example.csv'


def workflow(inchi, identity, file):
    try:
        bst_tp = pickle.load(open("model/UTP.dat" , "rb"))
        rf = pickle.load(open("model/DirectIO.dat", "rb"))
        bst_st = pickle.load(open("model/SPOTIC.dat" , "rb"))
            
        df_transporter = pd.read_csv(file)
        df_transporter = df_transporter[df_transporter['Sequence'].apply(lambda X:len(X) <1022)]
        df_transporter.reset_index(inplace = True, drop = True)
        with open("templates/candidates.fasta", 'w') as f:
            for ind in df_transporter.index:
                query = '>' + df_transporter['ID'][ind]
                seq = df_transporter['Sequence'][ind]
                f.write(query + '\n' + seq + '\n')
                           
        def calcualte_esm1b_vectors(df_transporter):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            model = model.to(device)
            batch_size=5
            batch_converter = alphabet.get_batch_converter()
            
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
        
        df_transporter = calcualte_esm1b_vectors(df_transporter)
        
        # identify transporter
        test_X = []
        for ind in df_transporter.index:
            emb = df_transporter["ESM1b"][ind]
            test_X.append(emb)
        features = ["ESM1b_" + str(i) for i in range(1280)]
        dtest = xgb.DMatrix(np.array(test_X), feature_names=features)
        y_test_pred = bst_tp.predict(dtest)
        df_transporter['UTP_Results'] = y_test_pred
                                
        # predict transport direction
        test_X = np.array(df_transporter["ESM1b"].tolist())
        probabilities = rf.predict_proba(test_X)
        probabilities_rounded = np.round(probabilities, 3)
        probabilities_df = pd.DataFrame(probabilities_rounded, columns=['Probability_Class_0', 'Probability_Class_1'])
        y_test_pred = probabilities_df['Probability_Class_1']
        df_transporter['DIOP_Results'] = y_test_pred

        # predict transporter and substrate
        df_transporter["InChI"] = ''
        for ind in df_transporter.index:
            df_transporter.loc[ind, 'InChI'] = inchi
            
        df_transporter["ECFP"] = ""
        for ind in df_transporter.index:
            mol = Chem.inchi.MolFromInchi(df_transporter["InChI"][ind])
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToBitString()
            df_transporter.loc[ind, "ECFP"] = ecfp
	

        test_X = []
        for ind in df_transporter.index:
            emb = df_transporter["ESM1b"][ind]
            ecfp = np.array(list(df_transporter["ECFP"][ind])).astype(int)
            test_X.append(np.concatenate([ecfp, emb]))
        test_X = np.array(test_X)
        
        feature_names = ["ECFP_" + str(i) for i in range(1024)]
        feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
        dtest = xgb.DMatrix(np.array(test_X), feature_names=feature_names)
        y_test_pred = bst_st.predict(dtest)
        df_transporter['SPOTIC_Results'] = y_test_pred

        del df_transporter["InChI"]
        del df_transporter["ECFP"]
        del df_transporter["ESM1b"]
                            
        # run blastp
        blastp_command = f'blastp -query templates/candidates.fasta -db ../ICT-DB/blastdb/ictdb -outfmt 6 -out -'
        result = subprocess.run(blastp_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            df = pd.read_csv(StringIO(result.stdout), sep='\t', header=None)
            df.columns = ["ID", "Subject ID", "%Identity", "Alignment Length","Mismatches", "Gap Openings", "Query Start", "Query End", "Subject Start", "Subject End", "E-value", "Bit Score"]
            df.sort_values(by='%Identity', ascending=False, inplace=True)
            top = df[df['%Identity'].apply(lambda x:x>=identity)]
            df_blastp = top.groupby('ID').apply(lambda group: [f"{row['Subject ID']} ({row['%Identity']})" for _, row in group.iterrows()]).reset_index(name='Blastp_Results')
            
            df_transporter = df_transporter.merge(df_blastp, on='ID', how='left')
            df_transporter.to_csv("TransDW_Result.csv", index=False)
            #os.remove('templates/candidates.fasta')
            
        else:
            print('Something went wrong when blastp!')
        	

    except Exception as e:
        print(f"An error occurred: {e}")
        
        
workflow(inchi, identity, file)
