# DeepMiR2GO
Introduction
------------

DeepMiR2GO is a novel tool that integrates three biological entities (microRNAs, proteins and diseases) information to automatlly annotate the Gene Ontology labels for microRNAs based on a deep hierarchical multi-label classification model. More specifically, DeepMiR2GO uses LINE(https://github.com/tangjianpku/LINE) to extract topological feature vectors of network and build a deep hierarchical classfication model referred to DeepGO (https://github.com/bio-ontology-research-group/deepgo). 

This repository contains script and data which were used to build and train the DeepMiR2GO model.

Data
----
* GOA_ensg_nonIEA_201604.pkl - The GO annotations of proteins of 201604 version with nonIEA evidence supported.
* MiR2GO_nonIEA_GOA_20180617.pkl - The GO annotations of microRNAs with nonIEA evidence supported.
* LINE_protein_embeddings_s100_n10_64.pkl/LINE_miRNA_embeddings_s100_n10_64.pkl - The embeddings of proteins and microRNAs extract from LINE with `samples = 100M`, `negative samples = 10`, `dimension = 64`.

Scripts4raw_data
-------
* txt2pkl.py/functions.pl scripts are used to prepare the raw data.

Hierarchical multi-label classification
-------------------------------
These scripts are modified from DeepGO. Refer to DeepGO (https://github.com/bio-ontology-research-group/deepgo) to install and run these scripts.
* hierarchical_classification.py - This script is used to build and train the multi-label classification model which uses network embeddings of proteins and microRNAs as an input.
* get_train-test_data.py - This script is used to prepare the train and test data.
* get_functions.py - This script is used to prepare the GO terms space of BP and MF.

Mutilabel classification_baseline
--------
Three classic multi-label classification methods: Decision Tree, Random Forest and Support Vector Machine, are used as baseline to compare with DeepMiR2GO to explore the classification performance.
* multiLabel_DT.py
* multiLabel_RF.py
* multiLabel_SVC.py
