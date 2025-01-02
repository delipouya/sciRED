import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sciRED import ensembleFCA as efca
from sciRED import glm
from sciRED import rotations as rot
from sciRED import metrics as met

from sciRED.utils import preprocess as proc
from sciRED.utils import visualize as vis
from sciRED.utils import corr
from sciRED.examples import ex_preprocess as exproc
from sciRED.examples import ex_visualize as exvis

from scipy.io import mmread
import scipy.sparse as sp
from scipy.sparse import load_npz
import random

np.random.seed(10)
NUM_COMPONENTS = 30
NUM_GENES = 2000
#NUM_GENES = 10000
NUM_COMP_TO_VIS = 5

metadata_path = "/home/delaram/sciRED//review_analysis/PBMC10K_3p5p_metadata_complete.csv"
matrix_path = "/home/delaram//sciRED/review_analysis/PBMC10K_3p5p_matrix.npz"
row_names_path = "/home/delaram//sciRED/review_analysis/PBMC10K_3p5p_row_names.csv"
col_names_path = "/home/delaram//sciRED/review_analysis/PBMC10K_3p5p_col_names.csv"

sparse_matrix = sp.load_npz(matrix_path)
data = sparse_matrix.toarray()
metadata = pd.read_csv(metadata_path)
row_names = pd.read_csv(row_names_path, header=None).squeeze()  # Row names (genes)
col_names = pd.read_csv(col_names_path, header=None).squeeze()  # Column names (cells)
genes = row_names[1:]
cells = col_names[1:] 

print("Matrix shape:", sparse_matrix.shape)  # Sparse matrix shape
print(data.shape)  # Print the shape of the sparse matrix
print(metadata.head())


### subset data based on non-zero gene sums and cell sums 
gene_sums = np.sum(data,axis=1) # row sums - library size
cell_sums = np.sum(data,axis=0) # col sums - sum reads in a gene
data = data[gene_sums != 0,:]
data = data[:,cell_sums != 0]

## subset the metadata based non-zero cell sums
metadata = metadata.loc[cell_sums != 0]

print(data.shape)

### calculate the variance for each gene
gene_vars = np.var(data, axis=1)
### select the top num_genes genes with the highest variance
gene_idx = np.argsort(gene_vars)[-NUM_GENES:]

#### select num_genes genes based on variance
## sort the gene_idx in ascending order
gene_idx = np.sort(gene_idx)
data = data[gene_idx,:]


### extract metadata sample_name as sample
y_sample = metadata['data']
y_celltype = metadata['predicted.id']
y_library_size = metadata['nCount_RNA']
num_genes, num_cells = data.shape

gene_sums = np.sum(data,axis=1) # row sums - library size
cell_sums = np.sum(data,axis=0) # col sums - sum reads in a gene

print(data.shape)
print(len(gene_sums))
print(len(cell_sums))

data = data.T #num cells, num genes

#### design matrix - library size only
x = y_library_size
## adding the intercept
x = sm.add_constant(x) ## adding the intercept

#### design matrix - library size and sample
column_levels = y_sample.unique() 
dict_covariate = {}
for column_level in column_levels:
    dict_covariate[column_level] = proc.get_binary_covariate(y_sample.squeeze(), column_level)
#### stack colummns of dict_covariate 
x_sample = np.column_stack(([dict_covariate[column] for column in column_levels]))

x = np.column_stack((y_library_size, x_sample)) 
x = sm.add_constant(x) ## adding the intercept

### fit GLM to each gene
glm_fit_dict = glm.poissonGLM(data, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
y = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y.shape) # (num_cells, num_genes)


####################################
#### Running PCA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=NUM_COMPONENTS))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_ 
pca_loading.shape #(factors, genes)

plt.plot(pca.explained_variance_ratio_)


my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
            for i in np.unique(y_celltype)}
cluster_color = [my_color[y_celltype.iloc[i]] for i in range(len(y_celltype))]

my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
            for i in np.unique(y_sample)}
sample_color = [my_color[y_sample.iloc[i]] for i in range(len(y_sample))]

#plt_legend_sample = exvis.get_legend_patch(y_sample, colors_dict_humanPBMC['sample'] )

### make a dictionary of colors for each sample in y_sample
vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
             cell_color_vec= cluster_color,
               legend_handles=False,
               title='PCA of 3p and 5p PBMC - cell type')

vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
             cell_color_vec= sample_color,
               legend_handles=False,
               title='PCA of 3p and 5p PBMC - sample')


#### plot the loadings of the factors
vis.plot_factor_loading(pca_loading.T, genes, 0, 1, fontsize=10, 
                    num_gene_labels=2,
                    title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)



####################################
#### Matching between factors and covariates ######
####################################

######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])


#make the title to two lines: 'varimax-PCA of pearson res- dorso lateral cortex data - cluster'
NUM_COMP_TO_VIS = 4
vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS, 
             cell_color_vec= cluster_color,
               legend_handles=False,
               title='varimax-PCA of pearson residuals\n 3p and 5p PBMC - cell type')

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS,
              cell_color_vec= sample_color,
                legend_handles=False,
                title='varimax-PCA of pearson residuals\n 3p and 5p PBMC - sample')

varimax_loading_df = pd.DataFrame(varimax_loading)
varimax_loading_df.columns = ['F'+str(i) for i in range(1, varimax_loading_df.shape[1]+1)]
varimax_loading_df.index = genes[gene_idx]


### save the varimax_loading_df and varimax_scores to a csv file
pca_scores_varimax_df = pd.DataFrame(pca_scores_varimax)
pca_scores_varimax_df.columns = ['F'+str(i) for i in range(1, pca_scores_varimax_df.shape[1]+1)]
if pca_scores_varimax_df.shape[0] == metadata.shape[0]:
    pca_scores_varimax_df_merged = pd.concat([metadata.reset_index(drop=True), pca_scores_varimax_df.reset_index(drop=True)], axis=1)
else:
    raise ValueError("Number of rows in 'metadata' does not match 'pca_scores_varimax_df'.")
pca_scores_varimax_df_merged.head()

### save the varimax_loading_df and varimax_scores to a csv file
varimax_loading_df.to_csv('/home/delaram/sciRED//review_analysis/varimax_loading_df_3p_5p_PBMC.csv')
pca_scores_varimax_df_merged.to_csv('/home/delaram/sciRED//review_analysis//pca_scores_varimax_df_3p_5p_PBMC.csv')


########################
######## PCA factors
factor_loading = pca_loading
factor_scores = pca_scores

##### Varimax factors
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax
covariate_vec = y_celltype
covariate_level = np.unique(covariate_vec)[1]

####################################
#### FCAT score calculation ######
####################################

### FCAT needs to be calculated for each covariate separately
fcat_sample = efca.FCAT(y_sample, factor_scores, scale='standard', mean='arithmatic')
fcat_cluster = efca.FCAT(y_celltype, factor_scores, scale='standard', mean='arithmatic')


### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_sample, fcat_cluster], axis=0)
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table

### using Otsu's method to calculate the threshold
threshold = efca.get_otsu_threshold(fcat.values.flatten())

vis.plot_histogram(fcat.values.flatten(),
                   xlabel='Factor-Covariate Association scores',
                   title='FCAT score distribution',
                   threshold=threshold)


## rownames of the FCAT table
all_covariate_levels = fcat.index.values
matched_factor_dist, percent_matched_fact = efca.get_percent_matched_factors(fcat, threshold)
matched_covariate_dist, percent_matched_cov = efca.get_percent_matched_covariates(fcat, threshold=threshold)
print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
vis.plot_matched_factor_dist(matched_factor_dist)
vis.plot_matched_covariate_dist(matched_covariate_dist, 
                                covariate_levels=all_covariate_levels)


### select the factors that are matched with any covariate level
matched_factor_index = np.where(matched_factor_dist>0)[0] 
fcat_matched = fcat.iloc[:,matched_factor_index] 
x_labels_matched = fcat_matched.columns.values
vis.plot_FCAT(fcat_matched, x_axis_label=x_labels_matched, title='', color='coolwarm',
                                 x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                                 x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, 
                                 save=False, save_path='../Plots/mean_importance_df_matched_3p_5p_PBMC.pdf')

factor_libsize_correlation = corr.get_factor_libsize_correlation(factor_scores, library_size = y_library_size)
vis.plot_factor_cor_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size', 
             y_label='Correlation', x_label='Factors')


### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_cluster], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table

vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
              x_axis_tick_fontsize=36, y_axis_tick_fontsize=40, 
              save=False, save_path='../Plots/mean_importance_df_matched_3p_5p_PBMC.pdf')


#cluster_fcat_sorted_scores, cluster_factors_sorted = vis.plot_sorted_factor_FCA_scores(fcat, 'cluster')

### select the factors that are matched with any covariate level
matched_factor_index = np.where(matched_factor_dist>0)[0] 
fcat_matched = fcat.iloc[:,matched_factor_index] 
x_labels_matched = fcat_matched.columns.values
vis.plot_FCAT(fcat_matched, x_axis_label=x_labels_matched, title='', color='coolwarm',
                                 x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                                 x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, 
                                 save=False, save_path='../Plots/mean_importance_df_matched_3p_5p_PBMC.pdf')


####################################
#### Bimodality scores
silhouette_score = met.kmeans_bimodal_score(factor_scores, time_eff=True)


bimodality_index = met.bimodality_index(factor_scores)
bimodality_score = bimodality_index

#bimodality_score = np.mean([silhouette_score, bimodality_index], axis=0)
#### Effect size
factor_variance = met.factor_variance(factor_scores)

## Specificity
simpson_fcat = met.simpson_diversity_index(fcat)

### label dependent factor metrics
asv_cell_type = met.average_scaled_var(factor_scores, covariate_vector=y_celltype, mean_type='arithmetic')
asv_sample = met.average_scaled_var(factor_scores, y_sample, mean_type='arithmetic')


########### create factor-interpretibility score table (FIST) ######
metrics_dict = {'Bimodality':bimodality_score, 
                    'Specificity':simpson_fcat,
                    'Effect size': factor_variance,
                    'Homogeneity (cell type)':asv_cell_type,
                    'Homogeneity (sample)':asv_sample}

fist = met.FIST(metrics_dict)
vis.plot_FIST(fist, title='Scaled metrics for all the factors')
### subset the first 15 factors of fist dataframe
vis.plot_FIST(fist.iloc[0:15,:])
vis.plot_FIST(fist.iloc[matched_factor_index,:])





