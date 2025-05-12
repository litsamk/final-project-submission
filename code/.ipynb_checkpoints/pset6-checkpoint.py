#Importing packages
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import anndata as ad
import scanpy as sc
import umap
import pydeseq2
from pydeseq2.preprocessing import deseq2_norm

#Loading the data and creating the samples and
data = pd.read_csv('../data/GSE217421_raw_counts_GRCh38.p13_NCBI.tsv', sep ='\t', index_col = 0).T
samples = pd.DataFrame(data.index, columns = ['Sample Name '])
expression = data.reset_index(drop = True)

#Filling out the samples info dataframe
sample_data = pd.read_excel('../data/Sample_reference.xlsx')
drug_data = pd.read_csv('../data/Drug_reference.txt', sep ='\t')
sample_drug_merge = pd.merge(sample_data, drug_data, left_on = 'Drug', right_on = 'Drug abbreviation')
final_sample_data = pd.merge(samples, sample_drug_merge, on = 'Sample Name ')
final_sample_data =final_sample_data.set_index('Sample Name ')

#Subsetting the expression data
relevant_samples = final_sample_data.index.tolist()
data_relevant = data[data.index.isin(relevant_samples)]
final_sample_data = final_sample_data.reindex(data_relevant.index)

#Normalizing the counts matrix (data_relevant)
data_normalized = deseq2_norm(data_relevant)[0]
size_factors = deseq2_norm(data_relevant)[1]

#Making the anndata objects
analysis_object = sc.AnnData(X = data_normalized, obs = final_sample_data)
analysis_object.layers['raw_counts'] = data_relevant
analysis_object.layers['normalized_counts'] = analysis_object.X.copy()
analysis_Anthracyclines = analysis_object[(analysis_object.obs['Drug class'] == 'Anthracycline') | (analysis_object.obs['Drug class'] == 'Control')].copy()

#running PCA
sc.pp.scale(analysis_Anthracyclines)
sc.tl.pca(analysis_Anthracyclines)
sc.pl.pca(analysis_Anthracyclines, components = [('1,2'), ('2,3')], color=['Drug name', 'Drug class', 'Cell line'], ncols = 2, wspace = 0.3, size = 100, save = '../outputs/Anthracyclines_PCA')
