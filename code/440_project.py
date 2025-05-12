# -*- coding: utf-8 -*-
"""
# Understanding Anthracycline Cardiotoxicity: Uncovering differential cardiotoxic pathways amongst chemotherapy drugs
Harshit Bhasin and Litsa Kapsalis
"""

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
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from mygene import MyGeneInfo
import gseapy as gp

"""# Preprocessing and data loading"""

#Loading the raw data and creating the samples dataframe
data = pd.read_csv('../data/GSE217421_raw_counts_GRCh38.p13_NCBI.tsv', sep ='\t', index_col = 0).T
samples = pd.DataFrame(data.index, columns = ['Sample Name '])
expression = data.reset_index(drop = True)

#Filling out the samples info dataframe
sample_data = pd.read_excel('../data/Sample_reference.xlsx')
drug_data = pd.read_csv('../data/Drug_reference.txt', sep ='\t')
# merge expression data with informationon on drug treatments
sample_drug_merge = pd.merge(sample_data, drug_data, left_on = 'Drug', right_on = 'Drug abbreviation')
final_sample_data = pd.merge(samples, sample_drug_merge, on = 'Sample Name ')
final_sample_data =final_sample_data.set_index('Sample Name ')

#Subsetting and aligning the expression data
relevant_samples = final_sample_data.index.tolist()
data_relevant = data[data.index.isin(relevant_samples)]
final_sample_data = final_sample_data.reindex(data_relevant.index)

#Normalizing the counts matrix (data_relevant) using deseq2
data_normalized = deseq2_norm(data_relevant)[0]
size_factors = deseq2_norm(data_relevant)[1]

#Making the anndata objects
analysis_object = sc.AnnData(X = data_normalized, obs = final_sample_data)
analysis_object.layers['raw_counts'] = data_relevant
analysis_object.layers['normalized_counts'] = analysis_object.X.copy()

# Separate out anthracyclines, the only drug treatment we'll be using for this analysis
analysis_Anthracyclines = analysis_object[(analysis_object.obs['Drug class'] == 'Anthracycline') | (analysis_object.obs['Drug class'] == 'Control')].copy()

#convert labels to list type
label_list = data_relevant.columns.tolist()

"""# Anthracyclines only





"""

#running PCA on the  anthracyclines only
sc.pp.scale(analysis_Anthracyclines)
sc.tl.pca(analysis_Anthracyclines)
sc.pl.pca(analysis_Anthracyclines, components = [('1,2'), ('2,3')], color=['Drug name', 'Drug class', 'Cell line'], ncols = 2, wspace = 0.3, size = 100)
plt.savefig('../outputs/PCAplots.png')

# Variance explained by each PC
analysis_Anthracyclines.uns['pca']['variance_ratio']

# Execute umap projections

#First, create a graph where each sample is a node and edges connect "neighbors"
# PCA transformed data is inputted
#Cosine similarity is used to compute nearest neighbors
sc.pp.neighbors(analysis_Anthracyclines, use_rep = 'X_pca', metric = 'cosine')

# Compute UMAP embedding
sc.tl.umap(analysis_Anthracyclines)
# Cluster with Leiden algorithm on the neighborhood graph
sc.tl.leiden(analysis_Anthracyclines, resolution=0.3)
# visualize the proejctions, displaying clustering by drug class (anthracyclines vs control), cell lines, individual drugs, and leiden clusters
sc.pl.umap(analysis_Anthracyclines, color = ['Drug class', 'Cell line', 'Drug name', 'leiden'], size = 100, ncols = 2, wspace = 0.3)
plt.savefig('../outputs/UMAPplots.png')

# Extract the leiden clusters and the drug names into a new df
drugname_df = analysis_Anthracyclines.obs[['leiden', 'Drug name']]
#create a new drug-leiden cluster distribution table
#Showing what percentage of samples for each drug are distributed into each cluster
pivot = pd.crosstab(drugname_df['Drug name'], drugname_df['leiden'], normalize='index') * 100

# Plot drug-cluster distributions as stacked bar plots
ax1 = pivot.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set1')
ax1.set_ylabel('Percentage of Samples')
ax1.legend(title='Leiden Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('../outputs/cluster_composition_drug_treatment.png')

# repeat the above for leiden clusters and cell lines
# Show how each cell line's samples are distributed across clusters
cellline_df = analysis_Anthracyclines.obs[['leiden', 'Cell line']]
pivot2 = pd.crosstab(cellline_df['Cell line'], cellline_df['leiden'], normalize='index') * 100
ax2 = pivot2.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set1')
ax2.set_ylabel('Percentage of Samples')
ax2.legend(title='Leiden Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('../outputs/cluster_composition_cell_line.png')

# Show which cell lines are represented within each cluster
pivot3 = pd.crosstab(cellline_df['leiden'], cellline_df['Cell line'], normalize='index') * 100
ax3 = pivot3.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set1')
ax3.set_ylabel('Percentage of Samples')
ax3.legend(title='Cell Line', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('../outputs/cell_line_cluster_division.png')

"""# Anthracycline deSeq"""

# Differentially Expressed Genes (DEG) analysis
# Execute deseq2 on the anthracyclines dataset

drug_name_and_leiden = DeseqDataSet(counts = analysis_Anthracyclines.layers['raw_counts'], metadata = analysis_Anthracyclines.obs, design_factors = ["`Drug name`", "leiden"])
drug_name_and_leiden.deseq2()

def diffexp(info, variable, condition1, condition2):
    """
    Perform differential expression analysis using DESeq2 results,
    annotate gene IDs with gene symbols, and filter significant results.

    Parameters:
    ----------
    info : DeseqDataSet
    variable : The name of the metadata column that we're grouping by
    condition1 : The reference condition/group to compare (e.g., 'control')
    condition2 : The test condition/group to compare against the reference
    Returns:
    -------
    stat_df_labeled : pd.DataFrame --> Full differential expression results with gene IDs and gene symbols.
    filtered : pd.DataFrame containing filtered results with baseMean ≥ 10.
    sigs : pd.DataFrame withstatistically significant and biologically meaningful DE genes
        (padj < 0.05 and |log2FoldChange| > 1)
    """

    # Run DESeq2 contrast between the two conditions
    stat = DeseqStats(info, contrast=(variable, condition1, condition2))
    stat.summary()

    stat_df = stat.results_df.copy()
    stat_df.insert(0, 'Geneid', label_list)
    # ADD GENE NAMES BASED ON GENE SYMBOL
    # Initialize MyGeneInfo
    mg = MyGeneInfo()
    # If your gene IDs are in the first column of df:
    gene_ids = stat_df.iloc[:, 0].tolist()
    # Query MyGene.info
    results = mg.querymany(gene_ids, scopes='entrezgene', fields='symbol', species='human')
   # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # Some IDs may not be mapped — filter out errors or NaNs
    results_df = results_df[['query', 'symbol']].dropna()
    # Rename columns for merging
    results_df.columns = ['Geneid', 'Symbol']
    stat_df[stat_df.columns[0]] = stat_df[stat_df.columns[0]].astype(str)
    results_df['Geneid'] = results_df['Geneid'].astype(str)
    # Now merge
    stat_df_labeled = stat_df.merge(results_df, left_on=stat_df.columns[0], right_on='Geneid', how='left')
    stat_df_labeled.dropna(subset = 'Symbol', inplace = True)
    filtered = stat_df_labeled[stat_df_labeled.baseMean >= 10]
    sigs = filtered[(filtered.padj < 0.05) & (abs(filtered.log2FoldChange) > 1)]

    return stat_df_labeled, filtered, sigs

# Run the diffexp() function to compare gene expression between  drugs vs control or drugs vs dox using DESeq2
# daun v dox
[results_daun_dox, filtered_daun_dox, sig_daun_dox] = diffexp(drug_name_and_leiden, 'Drug name', 'daunorubicin', 'doxorubicin')
# dox v ctrl
[results_dox, filtered_dox, sig_dox] = diffexp(drug_name_and_leiden, 'Drug name', 'doxorubicin', 'Control')
# daun v ctrl
[results_daun, filtered_daun, sig_daun] = diffexp(drug_name_and_leiden, 'Drug name', 'daunorubicin', 'Control')
# ida v ctrl
[results_ida, filtered_ida, sig_ida] = diffexp(drug_name_and_leiden, 'Drug name', 'idarubicin', 'Control')
# epi v ctrl
[results_epi, filtered_epi, sig_epi] = diffexp(drug_name_and_leiden, 'Drug name', 'epirubicin', 'Control')
# ida v dox
[results_ida_dox, filtered_ida_dox, sig_ida_dox] = diffexp(drug_name_and_leiden, 'Drug name', 'idarubicin', 'doxorubicin')
# epi v dox
[results_epi_dox, filtered_epi_dox, sig_epi_dox] = diffexp(drug_name_and_leiden, 'Drug name', 'epirubicin', 'doxorubicin')





#Making volcano plots and labeling them
#Daun v Dox
# Set thresholds
log2fc_thresh = 1
padj_thresh = 0.05

# Add a column for coloring
filtered_daun_dox['significance'] = 'Not significant'
filtered_daun_dox.loc[(filtered_daun_dox['log2FoldChange'] > log2fc_thresh) & (filtered_daun_dox['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_daun_dox.loc[(filtered_daun_dox['log2FoldChange'] < -log2fc_thresh) & (filtered_daun_dox['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_daun_dox = sig_daun_dox.loc[sig_daun_dox['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

# Create scatterplot daun vs dox DEGs
# Mark upregulated, down regulated, and non-significant genes
sns.scatterplot(
    data=filtered_daun_dox,
    x='log2FoldChange',
    y=-np.log10(filtered_daun_dox['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Daunorubicin v Doxorubicin')
plt.savefig('../outputs/DaunVDox_volcano.png')

# Create a heatmap of top ten DEGs for Daun v Dox
sns.heatmap(top10_daun_dox[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Daunorubicin v Doxorubicin')
plt.savefig('../outputs/DaunVDox_10DEGs.png')





filtered_daun_dox

# Repeat for ida v dox
# Add a column for coloring
filtered_ida_dox['significance'] = 'Not significant'
filtered_ida_dox.loc[(filtered_ida_dox['log2FoldChange'] > log2fc_thresh) & (filtered_ida_dox['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_ida_dox.loc[(filtered_ida_dox['log2FoldChange'] < -log2fc_thresh) & (filtered_ida_dox['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_ida_dox = sig_ida_dox.loc[sig_ida_dox['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

sns.scatterplot(
    data=filtered_ida_dox,
    x='log2FoldChange',
    y=-np.log10(filtered_ida_dox['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Idarubicin v Doxorubicin')
plt.savefig('../outputs/IdaVDox_volcano.png')
plt.figure(figsize = (8,6))
sns.heatmap(top10_ida_dox[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Idarubicin v Doxorubicin')
plt.savefig('../outputs/IdaVDox_10DEGs.png')

# Repeat for epi v dox
# Add a column for coloring
filtered_epi_dox['significance'] = 'Not significant'
filtered_epi_dox.loc[(filtered_epi_dox['log2FoldChange'] > log2fc_thresh) & (filtered_epi_dox['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_epi_dox.loc[(filtered_epi_dox['log2FoldChange'] < -log2fc_thresh) & (filtered_epi_dox['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_epi_dox = sig_epi_dox.loc[sig_epi_dox['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

sns.scatterplot(
    data=filtered_epi_dox,
    x='log2FoldChange',
    y=-np.log10(filtered_epi_dox['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
    alpha=0.6
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Epirubicin v Doxorubicin')
plt.savefig('../outputs/EpiVDox_volcano.png')
plt.figure(figsize = (8,6))
sns.heatmap(top10_epi_dox[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Epirubicin v Doxorubicin')
plt.savefig('../outputs/EpiVDox_10DEGs.png')

# Repeat for dox v control
# Add a column for coloring
filtered_dox['significance'] = 'Not significant'
filtered_dox.loc[(filtered_dox['log2FoldChange'] > log2fc_thresh) & (filtered_dox['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_dox.loc[(filtered_dox['log2FoldChange'] < -log2fc_thresh) & (filtered_dox['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_dox = sig_dox.loc[sig_dox['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

sns.scatterplot(
    data=filtered_dox,
    x='log2FoldChange',
    y=-np.log10(filtered_dox['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
    alpha=0.6
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Doxorubicin v Control')
plt.savefig('../outputs/DoxVCtrl_volcano.png')
plt.figure(figsize = (8,6))
sns.heatmap(top10_dox[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Doxorubicin v Control')
plt.savefig('../outputs/DoxVCtrl_10DEGs.png')

# Daun v control
# Add a column for coloring
filtered_daun['significance'] = 'Not significant'
filtered_daun.loc[(filtered_daun['log2FoldChange'] > log2fc_thresh) & (filtered_daun['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_daun.loc[(filtered_daun['log2FoldChange'] < -log2fc_thresh) & (filtered_daun['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_daun = sig_daun.loc[sig_daun['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

sns.scatterplot(
    data=filtered_daun,
    x='log2FoldChange',
    y=-np.log10(filtered_daun['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
    alpha=0.6
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Daunorubicin v Control')
plt.savefig('../outputs/DaunVCtrl_volcano.png')
plt.figure(figsize = (8,6))
sns.heatmap(top10_daun[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Daunorubicin v Control')
plt.savefig('../outputs/DaunVCtrl_10DEGs.png')

# Ida v control
# Add a column for coloring
filtered_ida['significance'] = 'Not significant'
filtered_ida.loc[(filtered_ida['log2FoldChange'] > log2fc_thresh) & (filtered_ida['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_ida.loc[(filtered_ida['log2FoldChange'] < -log2fc_thresh) & (filtered_ida['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_ida = sig_ida.loc[sig_ida['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

sns.scatterplot(
    data=filtered_ida,
    x='log2FoldChange',
    y=-np.log10(filtered_ida['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
    alpha=0.6
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Idarubicin v Control')
plt.savefig('../outputs/IdaVCtrl_volcano.png')
plt.figure(figsize = (8,6))
sns.heatmap(top10_ida[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Idarubicin v Control')
plt.savefig('../outputs/IdaVCtrl_10DEGs.png')

# Epi v control
# Add a column for coloring
filtered_epi['significance'] = 'Not significant'
filtered_epi.loc[(filtered_epi['log2FoldChange'] > log2fc_thresh) & (filtered_epi['padj'] < padj_thresh), 'significance'] = 'Upregulated'
filtered_epi.loc[(filtered_epi['log2FoldChange'] < -log2fc_thresh) & (filtered_epi['padj'] < padj_thresh), 'significance'] = 'Downregulated'

#Identify the top 10 DEGs
top10_epi = sig_epi.loc[sig_epi['log2FoldChange'].abs().sort_values(ascending=False).index].head(10)

sns.scatterplot(
    data=filtered_epi,
    x='log2FoldChange',
    y=-np.log10(filtered_epi['padj']),
    hue='significance',
    palette={'Upregulated': 'green', 'Downregulated': 'red', 'Not significant': 'gray'},
    alpha=0.6
)
plt.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1)
plt.axvline(-log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.axvline(log2fc_thresh, color='black', linestyle='--', linewidth=1)
plt.ylabel('-log10(padj)')
plt.title('Epirubicin v Control')
plt.savefig('../outputs/EpiVCtrl_volcano.png')
plt.figure(figsize = (8,6))
sns.heatmap(top10_epi[['log2FoldChange', 'Symbol']].set_index('Symbol'), linecolor = 'black', annot = True, cmap = 'viridis', linewidths = 0.75)
plt.title('Top 10 DEGs: Epirubicin v Control')
plt.savefig('../outputs/EpiVCtrl_10DEGs.png')

"""GSEA"""

def gsea_run(result_df):
  """
  Runs Gene Set Enrichment Analysis (GSEA) on a ranked list of genes.

  Parameters:
  result_df : A DataFrame containing differential expression results, including
        the columns 'Symbol' (gene names) and 'stat' ranking

  Returns:
  out : A DataFrame summarizing enriched gene sets from the GO_Biological_Process_2025 database.
   Columns include:
   - 'Term': Name of the enriched gene set
   - 'fdr': False discovery rate for the enrichment
   - 'es': Enrichment score
   - 'nes': Normalized enrichment score
    - 'pval': Raw p-value for enrichment
   - 'lead_genes': Genes driving the enrichment signal

  Function:
  - Filters and ranks genes by the 'stat' column, removes duplicate gene symbols
  - Runs GSEA using the gseapy.prerank method, 100 permutation
  - Collects and returns results sorted by FDR.

    Dependencies: gseapy package and access to GO_Biological_Process_2025
    """

  result_df_copy  = result_df.copy()
  ranking = result_df_copy [['Symbol', 'stat']].dropna().sort_values(by = 'stat', ascending = False)
  ranking.drop_duplicates(subset = ['Symbol'], inplace = True)
  pre_res = gp.prerank(rnk = ranking, gene_sets = 'GO_Biological_Process_2025', seed = 6, permutation_num = 100)
  out = []
  for item in list(pre_res.results):
    out.append([item, pre_res.results[item]['fdr'],
                        pre_res.results[item]['es'],
                        pre_res.results[item]['nes'],
                        pre_res.results[item]['pval'],
                        pre_res.results[item]['lead_genes']
                    ])

  out = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes', 'pval', 'lead_genes']).sort_values('fdr').reset_index(drop = True)
  return out

# Run GSEA on relevant comparisons (each drug vs dox or drug vs control)
gsea_daun_dox = gsea_run(results_daun_dox)
gsea_dox = gsea_run(results_dox)
gsea_daun = gsea_run(results_daun)
gsea_ida = gsea_run(results_ida)
gsea_epi = gsea_run(results_epi)
gsea_ida_dox = gsea_run(results_ida_dox)
gsea_epi_dox = gsea_run(results_epi_dox)

#Filter results by FDR
gsea_daun_dox_filtered = gsea_daun_dox[gsea_daun_dox.fdr < 0.05].sort_values(by ='nes')
gsea_dox_filtered = gsea_dox[gsea_dox.fdr < 0.05].sort_values(by ='nes')
gsea_daun_filtered = gsea_daun[gsea_daun.fdr < 0.05].sort_values(by ='nes')
gsea_ida_filtered = gsea_ida[gsea_ida.fdr < 0.05].sort_values(by ='nes')
gsea_epi_filtered = gsea_epi[gsea_epi.fdr < 0.05].sort_values(by ='nes')
gsea_ida_dox_filtered = gsea_ida_dox[gsea_ida_dox.fdr < 0.05].sort_values(by ='nes')
gsea_epi_dox_filtered = gsea_epi_dox[gsea_epi_dox.fdr < 0.05].sort_values(by ='nes')

#Making a heatmap of term enrichment from Dox v Ctrl terms in all drugs v ctrl

dox_terms_list =  gsea_dox_filtered.Term.to_list()
dox_terms_nes_dox = gsea_dox[gsea_dox['Term'].isin(dox_terms_list)][['Term', 'nes']].rename(columns = {'nes': 'Doxorubicin'})
dox_terms_nes_daun = gsea_daun[gsea_daun['Term'].isin(dox_terms_list)][['Term', 'nes']].rename(columns = {'nes': 'Daunorubicin'})
dox_terms_nes_ida = gsea_ida[gsea_ida['Term'].isin(dox_terms_list)][['Term', 'nes']].rename(columns = {'nes': 'Idarubicin'})
dox_terms_nes_epi = gsea_epi[gsea_epi['Term'].isin(dox_terms_list)][['Term', 'nes']].rename(columns = {'nes': 'Epirubicin'})
dox_terms_df = dox_terms_nes_dox.merge(dox_terms_nes_daun, on = 'Term', how = 'outer').merge(dox_terms_nes_epi, on = 'Term', how = 'outer').merge(dox_terms_nes_ida, on = 'Term', how = 'outer')
dox_terms_df.set_index('Term', inplace = True)

plt.figure(figsize=(10, 8))
sns.heatmap(dox_terms_df, linewidths = 0.1, annot = True)
plt.title('Normalized Enrichment Score v Control')
plt.savefig('../outputs/GSEA_anthracyclinesVCtrl.png')

# Visualized set of enriched gene sets in descending order
gsea_daun_dox_filtered.sort_values(by = 'nes', ascending = False)
daun_dox_terms = ['Mitochondrial Gene Expression (GO:0140053)', 'Proton Motive Force-Driven ATP Synthesis (GO:0015986)',
                  'tRNA Aminoacylation for Protein Translation (GO:0006418)',
                  'Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)', 'Aerobic Electron Transport Chain (GO:0019646)']
gsea_daun_dox_plot = gsea_daun_dox_filtered[gsea_daun_dox_filtered['Term'].isin(daun_dox_terms)]
plt.barh(gsea_daun_dox_plot['Term'], gsea_daun_dox_plot['nes'])
plt.xlabel('Normalized Enrichment Score')
plt.title('Enriched gene sets: Daunorubicin v Doxorubicin')
plt.savefig('../outputs/GSEA_DaunVDox.png')

# Repeat for selected genes from ida vs dox GSEA
gsea_ida_dox_filtered.sort_values(by = 'nes', ascending = False)
ida_dox_terms = ['Sarcomere Organization (GO:0045214)', 'Cardiac Muscle Cell Development (GO:0055013)',
                 'Inner Mitochondrial Membrane Organization (GO:0007007)', 'Mitochondrial Gene Expression (GO:0140053)',
                 'Oxidative Phosphorylation (GO:0006119)', 'Extracellular Matrix Organization (GO:0030198)',
                 'Collagen Fibril Organization (GO:0030199)', 'Extracellular Structure Organization (GO:0043062)',
                 'Proton Motive Force-Driven ATP Synthesis (GO:0015986)', 'Positive Regulation of Fibroblast Proliferation (GO:0048146)']

gsea_ida_dox_plot = gsea_ida_dox_filtered[gsea_ida_dox_filtered['Term'].isin(ida_dox_terms)]
plt.figure(figsize = (8,6))
plt.barh(gsea_ida_dox_plot['Term'], gsea_ida_dox_plot['nes'])
plt.xlabel('Normalized Enrichment Score')
plt.title('Enriched gene sets: Idarubicin v Doxorubicin')
plt.savefig('../outputs/GSEA_IdaVDox.png')

# Repeat for selected genes from epi vs dox GSEA

gsea_epi_dox_filtered.sort_values(by = 'nes', ascending = False)
epi_dox_terms = ['Oxidative Phosphorylation (GO:0006119)', 'Proton Motive Force-Driven Mitochondrial ATP Synthesis (GO:0042776)',
                 'Extracellular Structure Organization (GO:0043062)','Mitochondrial Translation (GO:0032543)',
                 'Nucleosome Organization (GO:0034728)', 'Protein Localization to Chromatin (GO:0071168)',
                 'Chromatin Organization (GO:0006325)', 'Heterochromatin Formation (GO:0031507)', 'Cardiac Conduction System Development (GO:0003161)']

gsea_epi_dox_plot = gsea_epi_dox_filtered[gsea_epi_dox_filtered['Term'].isin(epi_dox_terms)]
plt.figure(figsize = (8,6))
plt.barh(gsea_epi_dox_plot['Term'], gsea_epi_dox_plot['nes'])
plt.xlabel('Normalized Enrichment Score')
plt.title('Enriched gene sets: Epirubicin v Doxorubicin')
plt.savefig('../outputs/GSEA_EpiVDox.png')

# Repeat for selected genes from daun vs controls GSEA

gsea_daun_filtered.sort_values(by = 'nes', ascending = False)
daun_terms = ['Cardiac Muscle Cell Development (GO:0055013)', 'Striated Muscle Cell Development (GO:0055002)', 'Cardiac Myofibril Assembly (GO:0055003)',
              'Cardiac Muscle Cell Differentiation (GO:0055007)', 'Ventricular Cardiac Muscle Cell Action Potential (GO:0086005)']
gsea_daun_plot = gsea_daun_filtered[gsea_daun_filtered['Term'].isin(daun_terms)]
plt.barh(gsea_daun_plot['Term'], gsea_daun_plot['nes'])
plt.xlabel('Normalized Enrichment Score')
plt.title('Enriched gene sets: Daunorubicin v Controls')
plt.savefig('../outputs/GSEA_DaunVCtrl.png')

# Repeat for selected genes from ida vs controls GSEA

gsea_ida_filtered.sort_values(by = 'nes', ascending = False)
ida_terms = ['Extracellular Matrix Organization (GO:0030198)', 'Collagen Fibril Organization (GO:0030199)', 'DNA Synthesis Involved in DNA Repair (GO:0000731)',
             'Aerobic Electron Transport Chain (GO:0019646)', 'Oxidative Phosphorylation (GO:0006119)']
gsea_ida_plot = gsea_ida_filtered[gsea_ida_filtered['Term'].isin(ida_terms)]
plt.barh(gsea_ida_plot['Term'], gsea_ida_plot['nes'])
plt.xlabel('Normalized Enrichment Score')
plt.title('Enriched gene sets: Idarubicin v Controls')
plt.savefig('../outputs/GSEA_IdaVCtrl.png')

# Repeat for selected genes from epi vs controls GSEA

gsea_epi_filtered.sort_values(by = 'nes', ascending = False)
epi_terms = ['Double-Strand Break Repair (GO:0006302)', 'Protein Localization to Chromosome (GO:0034502)', 'Protein Localization to Centrosome (GO:0071539)',
                   'Cardiac Muscle Tissue Development (GO:0048738)', 'Regulation of Cardiac Muscle Cell Contraction (GO:0086004)', 'Cardiac Muscle Cell Development (GO:0055013)',
                   ]
gsea_epi_plot = gsea_epi_filtered[gsea_epi_filtered['Term'].isin(epi_terms)]
plt.barh(gsea_epi_plot['Term'], gsea_epi_plot['nes'])
plt.xlabel('Normalized Enrichment Score')
plt.title('Enriched gene sets: Epirubicin v Controls')
plt.savefig('../outputs/GSEA_EpiVCtrl.png')

#output all GSEA files
gsea_daun_dox_filtered.to_csv('../outputs/gsea_daun_dox_filtered.csv', index=False)
gsea_dox_filtered.to_csv('../outputs/gsea_dox_filtered.csv', index=False)
gsea_daun_filtered.to_csv('../outputs/gsea_daun_filtered.csv', index=False)
gsea_ida_filtered.to_csv('../outputs/gsea_ida_filtered.csv', index=False)
gsea_epi_filtered.to_csv('../outputs/gsea_epi_filtered.csv', index=False)
gsea_ida_dox_filtered.to_csv('../outputs/gsea_ida_dox_filtered.csv', index=False)
gsea_epi_dox_filtered.to_csv('../outputs/gsea_epi_dox_filtered.csv', index=False)

#output all DEG lists
sig_daun_dox.to_csv('../outputs/sig_daun_dox.csv', index=False)
sig_dox.to_csv('../outputs/sig_dox.csv', index=False)
sig_daun.to_csv('../outputs/sig_daun.csv', index=False)
sig_ida.to_csv('../outputs/ig_ida.csv', index=False)
sig_epi.to_csv('../outputs/sig_epi.csv', index=False)
sig_ida_dox.to_csv('../outputs/sig_ida_dox.csv', index=False)
sig_epi_dox.to_csv('../outputs/sig_epi_dox.csv', index=False)

def count_log2fc_direction(df):
    """Count positive/negative log2FoldChange in significant genes"""
    sigs = df[df['padj'] < 0.05]
    return {
        'positive': (sigs['log2FoldChange'] > 0).sum(),
        'negative': (sigs['log2FoldChange'] < 0).sum()
    }

dataframes = [sig_daun_dox, sig_dox, sig_daun, sig_ida, sig_epi, sig_ida_dox, sig_epi_dox]

variable_names = [
    name for df in dataframes
    for name, obj in globals().items()
    if obj is df and isinstance(obj, pd.DataFrame)
]

results = []
for name, df in zip(variable_names, dataframes):
    counts = count_log2fc_direction(df)
    results.append({
        'dataset': name,
        'positive_log2FC': counts['positive'],
        'negative_log2FC': counts['negative']
    })

results_df = pd.DataFrame(results)
print(results_df)