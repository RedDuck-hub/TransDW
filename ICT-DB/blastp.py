import sys
import os
import subprocess
import pandas as pd
from io import StringIO

def makeblastdb():
    makedb_command = f'makeblastdb -in ICTDB.fasta -dbtype prot -parse_seqids -out blastdb/ictdb'
    result = subprocess.run(makedb_command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("Database creation successful!")
    else:
        print("Error!")
        sys.exit(1)

def blast():
    blastp_command = f'blastp -query {blast_file} -db database/ictdb -outfmt 6 -out -'
    result = subprocess.run(blastp_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        df = pd.read_csv(StringIO(result.stdout), sep='\t', header=None)
        df.columns = [
            "Uniprot ID", "Subject ID", "%Identity", "Alignment Length",
            "Mismatches", "Gap Openings", "Query Start", "Query End",
            "Subject Start", "Subject End", "E-value", "Bit Score"
        ]
        
        df.sort_values(by=['Uniprot ID', '%Identity'], ascending=[True, False], inplace=True)
        top = df.groupby('Uniprot ID').head(4)
        result_df = top.groupby('Uniprot ID').apply(
            lambda group: [f"{row['Subject ID']} ({row['%Identity']})" for _, row in group.iterrows()]
        ).reset_index(name='Blastp_Results') 
        result_df.to_csv(f'{db_name}_blastp_result.csv', index=False)
        print("Sequence alignment successful!")

    else:
        print("Error!")
        sys.exit(1)
        
        
if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)
    blast_file = ''
    
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-i':
            blast_file = sys.argv[i+1]
                            
    if not os.path.isfile(blast_file):
        print("File not found!")
        sys.exit(1)    

    makeblastdb()
    blast()

    

