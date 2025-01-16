import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import pyuniprot
import time
import requests
import os

def UniprotCaller():
    df = pd.read_csv('ICTDB.csv')
    targets = []
    for ind in df.index:
        if pd.isnull(df['function'][ind]):
            targets.append(df['uniprot_id'][ind])
         
    targets = list(set(targets))
    
    def get(targets):
        for target in tqdm(targets):
            url = f'https://rest.uniprot.org/uniprotkb/{target}.txt'
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    lines = response.text.splitlines()
                    func = ''
                    for line in lines:
                   
                        if line.startswith('CC'):
                            if line.startswith('CC   -!- INTERACTION') or line.startswith('CC   -!- ACTIVITY') or line.startswith('CC   -!- CATALYTIC') or line.startswith('CC   -!- SUBCELLULAR') or line.startswith('CC   -!- SUBCELLULAR') or line.startswith('CC   -!- SIMILARITY') or line.startswith('CC   -!- BIOPHYSICOCHEMICAL') or line.startswith('CC   -!- PATHWAY') or line.startswith('CC   -!- SUBUNIT') or line.startswith('CC   -!- INDUCTION') or line.startswith('CC   -!- DISRUPTION') or line.startswith('CC   -!- SIMILARITY'):        
                                break
                            func += line.replace('CC   -!- FUNCTION: ', '').replace('   ', '')
                            func += line.replace('CC   -!- FUNCTION: ', '').replace('   ', '')
                            
                      
                    if func:
                        target_rows = df[df['uniprot_id'] == target]
                        
                        for ind in target_rows.index:
                            df.loc[ind, 'function'] = func
            
            except:
                print(f'error! {target} not find function')
            
            finally:
                time.sleep(1)
            
    get(targets)
    df.to_csv('ICTDB.csv', index=False)
    

    targets = []
    for ind in df.index:
        if pd.isnull(df['p_name'][ind]):
            targets.append(df['uniprot_id'][ind])
         
    targets = list(set(targets))
    
    def get(targets):
        for target in tqdm(targets):
            url = f'https://rest.uniprot.org/uniprotkb/{target}.txt'
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    lines = response.text.splitlines()
                    name = ''
                    for line in lines:
                        if line.startswith('DE   RecName: Full='):
                            name = line.replace('DE   RecName: Full=', '')
                            if '{' in name:
                                name = name[: name.find('{')]
                            if ';' in name:
                                name = name[: name.find(';')]
 
                            
                      
                    if name:
                        target_rows = df[df['uniprot_id'] == target]
                        
                        for ind in target_rows.index:
                            df.loc[ind, 'p_name'] = name
            
            except:
                print(f'error! {target} not find name')
            
            finally:
                time.sleep(1)
            
    get(targets)
    df.to_csv('ICTDB.csv', index=False)
    
    
    targets = []
    for ind in df.index:
        if pd.isnull(df['lineage'][ind]):
            targets.append(df['uniprot_id'][ind])
         
    targets = list(set(targets))
    
    def get(targets):
        for target in tqdm(targets):
            url = f'https://rest.uniprot.org/uniprotkb/{target}.txt'
            
            response = requests.get(url)
            try:
                if response.status_code == 200:
                    lines = response.text.splitlines()
                    lineage = ''
                    for line in lines:
                        if line.startswith('OC'):
                            lineage += line.replace('OC', '').replace(' ', '')
                        
                    if lineage:
                        target_rows = df[df['uniprot_id'] == target]
                        
                        for ind in target_rows.index:
                            df.loc[ind, 'lineage'] = lineage
            
            except:
                print(f'error! {target} not find lineage')
            
            finally:
                time.sleep(1)
    
    get(targets)
    df.to_csv('ICTDB.csv', index=False)
    
    
    targets = []
    for ind in df.index:
        if pd.isnull(df['p_structure'][ind]):
            targets.append(df['uniprot_id'][ind])
         
    targets = list(set(targets))
    
    def get(targets):
        for target in tqdm(targets):
            structure_link = ''
            if os.path.exists('C:\\Users\\Redduck\\Desktop\\Result\\Web\\static\\ICTDB\\Transporters\\{target}.pdb'):
                structure_link = f'C:\\Users\\Redduck\\Desktop\\Result\\Web\\static\\ICTDB\\Transporters\\{target}.pdb'
                target_rows = df[df['uniprot_id'] == target]
                for ind in target_rows.index:
                    df.loc[ind, 'p_structure'] = structure_link
                
            else:
                url = f'https://alphafold.ebi.ac.uk/files/AF-{target}-F1-model_v4.pdb'
                response = requests.get(url)
                
                try:
                    if response.status_code == 200:
                        with open(f'C:\\Users\\Redduck\\Desktop\\Result\\Web\\static\\ICTDB\\Transporters\\{target}.pdb', 'wb') as f:
                            f.write(response.content)
                        
                        structure_link = f'C:\\Users\\Redduck\\Desktop\\Result\\Web\\static\\ICTDB\\Transporters\\{target}.pdb'
                        target_rows = df[df['uniprot_id'] == target]
                        for ind in target_rows.index:
                            df.loc[ind, 'p_structure'] = structure_link
                                             
                except:        
                    print(f'Failed to download {pdb}')
                    
                finally:
                    time.sleep(2)
            
    get(targets)
    df.to_csv('ICTDB.csv', index=False)
    
def csv_to_fasta():
    df = pd.read_csv('ICTDB.csv')
    print(len(df))
    df = df.drop_duplicates(subset='uniprot_id', keep='first')
    print(len(df))
    with open('ICTDB.fasta', 'w') as f:
        for ind in df.index:
            query = '>' + df['uniprot_id'][ind]
            seq = df['sequence'][ind]
            f.write(query + '\n' + seq + '\n')