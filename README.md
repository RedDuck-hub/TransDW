# TransDW: a mult-scale transporter discovery workflow

Efficient transporters are essential for high yields in industrial microbial production. However, the limited number of functionally characterized transporters emphasizes the requirement for effective identification methods. Machine learning has advanced predictions of transporters and their substrate specificities. 

TransDW, a novel multi-scale workflow for transporter discovery, which integrates three ML models and a database: 1) UTP, which identifies whether undefined proteins belong to transporters. 2) DirectIO, which predicts the transport direction of transporters, whether inward or outward. 3) SPOTIC, which predicts specific substrates for transporters. 4) ICT-DB, which contains a manually curated collection of 1675 industrial compounds and their associated transporters. Multiple sequence alignment (MSA) allows for the identification of functionally similar transporters for candidates.

![](https://mtc123.oss-cn-beijing.aliyuncs.com/工作内网/郝锦康/日常计算/郝锦康/2025-01-16-15-49-22-graphical_abstract.tif)



## Environment setup

TransDW is developed based on Python 3.8. Please run the following code to import the conda environment:

```
conda env create -f environment.yml
```



## Model training and prediction

Here, we provide the training and usage code for three models in TransDW. Some data is not included due to its large storage size; please download it from Zenodo: https://zenodo.org/records/14666289



## Workflow execution

In the **Workflow** folder, code is provided to run UTP, DirectIO, SPOTIC, and Blastp simultaneously. The program accepts three parameters as input:

- InChI of the substrate: Used for the specific substrate prediction of transporters. Multiple transporter candidates can be accepted in a single calculation, but only one substrate can be provided.
- Sequence identity threshold: Only proteins with sequence similarity to the candidate exceeding this threshold will be included in the Blastp search results.
- Transporter input file: Contains one or more transporters to be predicted. The file format must be CSV and must include the columns "ID" and "Sequence", please refer to ***Help*** for details.

Temporary files generated during execution are saved by default in the **templates** folder in the same directory. Do not delete these files before the program finishes running. The final calculation results will include prediction scores from the three models as well as the Blastp search results.



## Help

- The operation of TransDW requires specific versions of Python libraries to ensure the correct functioning of the program. We recommend using the anaconda environment configuration provided here;
- Please ensure that the column names in the input files of the program are correct. For example, the protein ID can be represented by Uniprot ID or accession number, but not by pure numbers. There is no restriction on the choice of substrate names. For more details, please refer to the example files in each folder;
- TransDW utilizes the pre-trained ESM1b model to embed protein sequences. When using the ESM1b model for the first time, its pre-trained configuration files will be automatically downloaded to the local computing environment.
