import numpy as np
import pandas as pd
import pickle
import esm
import torch
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

model_file = 'DIOP.dat'
predict_file = 'example_diop.csv'
result_file = 'DIOP_result.csv'

try:
    with open(model_file, "rb") as f:
        rf = pickle.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    def convert_vectors(df_transporter, batch_size=5):
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

    transporters = pd.read_csv(predict_file)
    transporters = convert_vectors(transporters)
    
    # building a dataset
    X_test = np.array(transporters["ESM1b"].tolist())
    probabilities = rf.predict_proba(X_test)
    probabilities_rounded = np.round(probabilities, 4)
    probabilities_df = pd.DataFrame(probabilities_rounded, columns=['Probability_Class_0', 'Probability_Class_1'])
    y_test_pred = probabilities_df['Probability_Class_1']
    
    # generating results  
    transporters['Result'] = ''
    for ind, result in zip(transporters.index, y_test_pred):
        transporters.loc[ind, 'Result'] = result
        
    del transporters['ESM1b']
    transporters.to_csv(result_file, index=False)

except Exception as e:
    print(f"An error occurred: {e}")
