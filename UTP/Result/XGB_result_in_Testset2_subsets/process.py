import pandas as pd
import os

file_list = os.listdir()
file_list.remove('process.py')
datas = {}
datas['Subset'] = []
datas['Total'] = []
datas['Yes'] = []
datas['No'] = []
for f in file_list:
    df = pd.read_csv(f)
    subset = df['Strain subset'][1]

    df = df[df['Label'].apply(lambda x: x==1)]
    total = len(df)
    
    df = df[df['XGB_Prediction'].apply(lambda x: x is True)]
    yes = len(df)
    no = total - yes
    
    datas['Subset'].append(subset)
    datas['Total'].append(total)
    datas['Yes'].append(yes)
    datas['No'].append(no)
    
    
df = pd.DataFrame(datas)
df.to_csv('barplot.csv', index=False)