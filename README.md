# TransDW: a mult-scale transporter discovery workflow

Efficient transporters are essential for high yields in industrial microbial production. However, the limited number of functionally characterized transporters emphasizes the requirement for effective identification methods. Machine learning has advanced predictions of transporters and their substrate specificities. 

TransDW, a novel multi-scale workflow for transporter discovery, which integrates three ML models and a database: 1) UTP, which identifies whether undefined proteins belong to transporters. 2) DirectIO, which predicts the transport direction of transporters, whether inward or outward. 3) SPOTIC, which predicts specific substrates for transporters. 4) ICT-DB, which contains a manually curated collection of 1675 industrial compounds and their associated transporters. Multiple sequence alignment (MSA) allows for the identification of functionally similar transporters for candidates.

![](https://mtc123.oss-cn-beijing.aliyuncs.com/工作内网/郝锦康/日常计算/郝锦康/2025-01-16-15-49-22-graphical_abstract.tif)



## Environment Setup

TransDW is developed based on Python 3.8. Please run the following code to import the conda environment:

```
conda env create -f environment.yml
```
