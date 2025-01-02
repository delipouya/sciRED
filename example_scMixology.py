import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import ssl; ssl._create_default_https_context = ssl._create_unverified_context
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
import time

np.random.seed(10)
NUM_COMPONENTS = 10
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5

data_file_path = '/home/delaram/sciRED/Data/scMix_3cl_merged.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)
y_cell_line, y_sample, y_protocol = exproc.get_metadata_scMix(data)
data.obs['protocol'] = y_protocol.to_numpy()
data.obs['cell_line'] = y_cell_line.to_numpy()
data.obs['sample'] = y_sample.to_numpy()

colors_dict_scMix = exvis.get_colors_dict_scMix(y_protocol, y_cell_line)
plt_legend_cell_line = exvis.get_legend_patch(y_cell_line, colors_dict_scMix['cell_line'] )
plt_legend_protocol = exvis.get_legend_patch(y_sample, colors_dict_scMix['protocol'] )


############## log normalizing the data (optional - for comparison)
y = np.log(y+1e-10)

####################################
#### fit GLM to each gene ######
####################################

#### Design-1: design matrix - library size only
x = proc.get_library_design_mat(data, lib_size='nCount_originalexp')

#### Design-2: design matrix - library size and sample
x_protocol = proc.get_design_mat(metadata_col='protocol', data=data) 
x = np.column_stack((data.obs.nCount_originalexp, x_protocol)) 
x = sm.add_constant(x) ## adding the intercept


start = time.time()
glm_fit_dict = glm.poissonGLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
y = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y.shape) # (num_cells, num_genes)

################################################
#### Running PCA on the pearson residual ######
################################################

### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=NUM_COMPONENTS))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape
#plt.plot(pca.explained_variance_ratio_)

end = time.time()
print('Time taken: ', end - start)
print('time taken (min): ', (end - start)/60) 


title = 'PCA of pearson residuals - lib size/protocol removed'
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)

### make a dictionary of colors for each sample in y_sample
vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_protocol)


#### plot the loadings of the factors
vis.plot_factor_loading(pca_loading.T, genes, 0, 2, fontsize=10, 
                    num_gene_labels=2,
                    title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)

vis.plot_umap(pca_scores, 
              title='UMAP',
              cell_color_vec= colors_dict_scMix['protocol'] , 
               legend_handles=True,plt_legend_list=plt_legend_protocol)


vis.plot_umap(pca_scores, 
              title='UMAP',
              cell_color_vec= colors_dict_scMix['cell_line'] , 
               legend_handles=True,plt_legend_list=plt_legend_cell_line)



######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

title = 'Varimax PCA of pearson residuals '

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_protocol)

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)


######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax(pca_loading.T)
promax_loading = rotation_results_promax['rotloading']
pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
vis.plot_pca(pca_scores_promax, 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title='Promax PCA of pearson residuals ',
               plt_legend_list=plt_legend_protocol)

vis.plot_pca(pca_scores_promax, 4, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title='Promax PCA of pearson residuals ',
               plt_legend_list=plt_legend_cell_line)


#### plot the loadings of the factors
vis.plot_factor_loading(varimax_loading, genes, 0, 4, fontsize=10, 
                    num_gene_labels=6,title='Scatter plot of the loading vectors', 
                    label_x=False, label_y=False)


### calculate the correlation between each factor of varimax and promax factor scores
factor_corr = np.zeros((pca_scores_varimax.shape[1], pca_scores_promax.shape[1]))
for i in range(pca_scores_varimax.shape[1]):
    for j in range(pca_scores_promax.shape[1]):
        factor_corr[i,j] = np.corrcoef(pca_scores_varimax[:,i], pca_scores_promax[:,j])[0,1]
factor_corr_df = pd.DataFrame(factor_corr)
factor_corr_df.index = ['F'+str(i+1) for i in range(pca_scores_varimax.shape[1])]
factor_corr_df.columns = ['F'+str(i+1) for i in range(pca_scores_promax.shape[1])]
factor_corr_df.head()
factor_corr_df = factor_corr_df.iloc[0:15,0:15]

plt.figure(figsize=(15,12))
plt.imshow(factor_corr_df, cmap='coolwarm')
plt.xticks(np.arange(factor_corr_df.shape[1]), factor_corr_df.columns.values, rotation=90, fontsize=30)
plt.yticks(np.arange(factor_corr_df.shape[0]), factor_corr_df.index.values, fontsize=30)
plt.xlabel('Promax factors', fontsize=34)
plt.ylabel('Varimax factors', fontsize=34)
plt.title('Correlation between varimax and promax factors', fontsize=34)
plt.colorbar()
plt.show()


### calculate the correlation between varimax and promax factors 0 to 30 (diagnoal of the factor_corr_df)
factor_corr_diag = np.zeros(pca_scores_varimax.shape[1])
for i in range(pca_scores_varimax.shape[1]):
    factor_corr_diag[i] = np.corrcoef(pca_scores_varimax[:,i], pca_scores_promax[:,i])[0,1]
plt.figure(figsize=(10,5))
plt.hist(factor_corr_diag, bins=20)
plt.xlabel('Correlation between varimax and promax factors', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Histogram of the correlation between varimax and promax factors', fontsize=20)
plt.show()


### calculate the correlation between each factor of varimax and pca factor scores
factor_corr = np.zeros((pca_scores_varimax.shape[1], pca_scores.shape[1]))
for i in range(pca_scores_varimax.shape[1]):
    for j in range(pca_scores.shape[1]):
        factor_corr[i,j] = np.corrcoef(pca_scores_varimax[:,i], pca_scores[:,j])[0,1]
factor_corr_df = pd.DataFrame(factor_corr)
factor_corr_df.index = ['F'+str(i+1) for i in range(pca_scores_varimax.shape[1])]
factor_corr_df.columns = ['F'+str(i+1) for i in range(pca_scores.shape[1])]
factor_corr_df = factor_corr_df.iloc[0:15,0:15]


plt.figure(figsize=(15,12))
plt.imshow(factor_corr_df, cmap='coolwarm')
plt.xticks(np.arange(factor_corr_df.shape[1]), factor_corr_df.columns.values, rotation=90, fontsize=30)
plt.yticks(np.arange(factor_corr_df.shape[0]), factor_corr_df.index.values, fontsize=30)
plt.xlabel('PCA factors', fontsize=34)
plt.ylabel('Varimax factors', fontsize=34)
plt.title('Correlation between varimax and PCA factors', fontsize=34)
plt.colorbar()
plt.show()

###################################################
####  Factors and covariate association ######
###################################################

#### Experiment-1: PCA factors
factor_loading = pca_loading
factor_scores = pca_scores

##### Experiment-2: Varimax factors
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax

##### Experiment-3: Promax factors
factor_loading = rotation_results_promax['rotloading']
factor_scores = pca_scores_promax




#### save varimax loading and scores to a csv file
varimax_loading_df = pd.DataFrame(varimax_loading)
varimax_loading_df.columns = ['F'+str(i) for i in range(1, varimax_loading_df.shape[1]+1)]
varimax_loading_df.index = genes
varimax_loading_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/sciRED_loading_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')

pca_scores_varimax_df = pd.DataFrame(pca_scores_varimax)
pca_scores_varimax_df.columns = ['F'+str(i) for i in range(1, pca_scores_varimax_df.shape[1]+1)]
pca_scores_varimax_df.index = data.obs.index.values
pca_scores_varimax_df_merged = pd.concat([data.obs.reset_index(drop=True), pca_scores_varimax_df.reset_index(drop=True)], axis=1)
pca_scores_varimax_df_merged.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/sciRED_scores_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')


####################################
#### FCAT score calculation ######
####################################

### FCAT needs to be calculated for each covariate separately
fcat_protocol = efca.FCAT(y_protocol, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line, factor_scores, scale='standard', mean='arithmatic')

### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)
fcat.shape
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

### visualize the first 15 factors
vis.plot_FCAT(fcat.iloc[:,0:15],title='', color='coolwarm',x_axis_fontsize=35, 
              y_axis_fontsize=35, title_fontsize=35,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

## rownames of the FCAT table
all_covariate_levels = fcat.index.values

vis.plot_histogram(fcat.values.flatten(), 
                   xlabel='Factor-Covariate Association scores',
                     title='FCAT score distribution') 

### using Otsu's method to calculate the threshold
threshold = efca.get_otsu_threshold(fcat.values.flatten())

vis.plot_histogram(fcat.values.flatten(),
                   xlabel='Factor-Covariate Association scores',
                   title='FCAT score distribution',
                   threshold=threshold)

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
vis.plot_FCAT(fcat_matched, x_axis_label=x_labels_matched, title='', color='coolwarm')


factor_libsize_correlation = corr.get_factor_libsize_correlation(factor_scores, library_size = data.obs.nCount_originalexp)
vis.plot_factor_cor_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size', 
             y_label='Correlation', x_label='Factors')


####################################
#### Bimodality scores
silhouette_score = met.kmeans_bimodal_score(factor_scores, time_eff=True)
bimodality_index = met.bimodality_index(factor_scores)
bimodality_score = np.mean([silhouette_score, bimodality_index], axis=0)

#### Effect size
factor_variance = met.factor_variance(factor_scores)

## Specificity
simpson_fcat = met.simpson_diversity_index(fcat)

### label dependent factor metrics
asv_cell_line = met.average_scaled_var(factor_scores, covariate_vector=y_cell_line, mean_type='arithmetic')
asv_sample = met.average_scaled_var(factor_scores, y_sample, mean_type='arithmetic')

#### plot the ralative variance table
svt_cell_line = met.scaled_var_table(factor_scores, y_cell_line)
svt_protocol = met.scaled_var_table(factor_scores, y_protocol)
svt = pd.concat([svt_cell_line, svt_protocol], axis=0)
vis.plot_relativeVar(svt, title='Relative variance score table')


########### create factor-interpretibility score table (FIST) ######
metrics_dict = {'Bimodality':bimodality_score, 
                    'Specificity':simpson_fcat,
                    'Effect size': factor_variance,
                    'Homogeneity (cell line)':asv_cell_line,
                    'Homogeneity (protocol)':asv_sample}

fist = met.FIST(metrics_dict)
vis.plot_FIST(fist, title='Scaled metrics for all the factors')
### subset the first 15 factors of fist dataframe
vis.plot_FIST(fist.iloc[0:15,:])
vis.plot_FIST(fist.iloc[matched_factor_index,:])
