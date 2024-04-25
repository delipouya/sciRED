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
from sciRED.utils import preprocess as proc
from sciRED.utils import visualize as vis
from exutils import ex_preprocess as exproc
from exutils import ex_visualize as exvis

np.random.seed(10)
NUM_COMPONENTS = 30
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5

data_file_path = '/home/delaram/sciFA/Data/scMix_3cl_merged.h5ad'
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


############## log normalizing the data
y = np.log(y+1e-10)

####################################
#### fit GLM to each gene ######
####################################

#### design matrix - library size only
x = proc.get_library_design_mat(data, lib_size='nCount_originalexp')

#### design matrix - library size and sample
x_protocol = proc.get_design_mat('protocol', data) 
x = np.column_stack((data.obs.nCount_originalexp, x_protocol)) 
x = sm.add_constant(x) ## adding the intercept

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
plt.plot(pca.explained_variance_ratio_)


title = 'PCA of pearson residuals - reg: lib size/protocol'
title = ''
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
                    label_x=False, label_y=False)

vis.plot_umap(pca_scores, colors_dict_scMix['protocol'] , covariate='protocol', title='UMAP')
vis.plot_umap(pca_scores, colors_dict_scMix['cell_line'] , covariate='cell_line',title='UMAP')


###################################################
#### Matching between factors and covariates ######
###################################################
covariate_name = 'cell_line'#'cell_line'
#factor_scores = pca_scores
covariate_vector = y_cell_line
y_cell_line.unique()
covariate_level = 'HCC827'


######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

title = 'Varimax PCA of pearson residuals '
num_pc=5
vis.plot_pca(pca_scores_varimax, num_pc, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_protocol)

vis.plot_pca(pca_scores_varimax, num_pc, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)


######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax_rotation(pca_loading.T)
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
### set the row and column names of factor_corr_df as 'F1', 'F2', ...
factor_corr_df.index = ['F'+str(i+1) for i in range(pca_scores_varimax.shape[1])]
factor_corr_df.columns = ['F'+str(i+1) for i in range(pca_scores_promax.shape[1])]
factor_corr_df.head()
factor_corr_df = factor_corr_df.iloc[0:15,0:15]
### visualize teh factor_corr_df as a heatmap without a function
plt.figure(figsize=(15,12))
### visualize the factor_corr_df factors 1 to 15
plt.imshow(factor_corr_df, cmap='coolwarm')
## set the x and y axis ticks labels as F1, F2, ...
plt.xticks(np.arange(factor_corr_df.shape[1]), factor_corr_df.columns.values, rotation=90, fontsize=30)
plt.yticks(np.arange(factor_corr_df.shape[0]), factor_corr_df.index.values, fontsize=30)

plt.xlabel('Promax factors', fontsize=34)
plt.ylabel('Varimax factors', fontsize=34)
plt.title('Correlation between varimax and promax factors', fontsize=34)
plt.show()

### calculate the correlation between varimax and promax factors 0 to 30 (diagnoal of the factor_corr_df)
factor_corr_diag = np.zeros(pca_scores_varimax.shape[1])
for i in range(pca_scores_varimax.shape[1]):
    factor_corr_diag[i] = np.corrcoef(pca_scores_varimax[:,i], pca_scores_promax[:,i])[0,1]
### show the histogram of the factor_corr_diag
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
### set the row and column names of factor_corr_df as 'F1', 'F2', ...
factor_corr_df.index = ['F'+str(i+1) for i in range(pca_scores_varimax.shape[1])]
factor_corr_df.columns = ['F'+str(i+1) for i in range(pca_scores.shape[1])]
factor_corr_df.head()
factor_corr_df = factor_corr_df.iloc[0:15,0:15]
### visualize teh factor_corr_df as a heatmap without a function
plt.figure(figsize=(15,12))
### visualize the factor_corr_df factors 1 to 15
plt.imshow(factor_corr_df, cmap='coolwarm')
## set the x and y axis ticks labels as F1, F2, ...
plt.xticks(np.arange(factor_corr_df.shape[1]), factor_corr_df.columns.values, rotation=90, fontsize=30)
plt.yticks(np.arange(factor_corr_df.shape[0]), factor_corr_df.index.values, fontsize=30)
plt.xlabel('PCA factors', fontsize=34)
plt.ylabel('Varimax factors', fontsize=34)
plt.title('Correlation between varimax and PCA factors', fontsize=34)


####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'protocol'#'cell_line'
covariate_level = b'Dropseq'
covariate_vector = y_protocol

########################
factor_loading = pca_loading
factor_scores = pca_scores
########################


########################
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax


########################
factor_loading = rotation_results_promax['rotloading']
factor_scores = pca_scores_promax


####################################
#### Mean Importance score
####################################

### calculate the mean importance of each covariate level
mean_importance_df_protocol = efca.get_mean_importance_all_levels(y_protocol, factor_scores, scale='standard', mean='arithmatic')
mean_importance_df_cell_line = efca.get_mean_importance_all_levels(y_cell_line, factor_scores, scale='standard', mean='arithmatic')

### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
mean_importance_df = pd.concat([mean_importance_df_protocol, mean_importance_df_cell_line], axis=0)
mean_importance_df.shape
vis.FCAT(mean_importance_df, title='F-C Match: Feature importance scores', 
                                 color='coolwarm',x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
                               x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

### only visualize teh first 15 factors
fplot.plot_all_factors_levels_df(mean_importance_df.iloc[:,0:15],
                                    title='', 
                                    color='coolwarm',x_axis_fontsize=35, y_axis_fontsize=35, title_fontsize=35,
                                x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values

##### Define global metrics for how well a factor analysis works on a dataset 
#### given a threshold for the feature importance scores, calculate the percentage of the factors that are matched with any covariate level
### plot the histogram of all the values in mean importance scores
fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                     title='F-C Match: Feature importance scores') 

### choosing a threshold for the feature importance scores

threshold = 0.3
threshold = efca.get_otsu_threshold(mean_importance_df.values.flatten())

fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                        title='F-C Match: Feature importance scores', threshold=threshold)

matched_factor_dist, percent_matched_fact = efca.get_percent_matched_factors(mean_importance_df, threshold)
matched_covariate_dist, percent_matched_cov = efca.get_percent_matched_covariate(mean_importance_df, threshold=threshold)

print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
fplot.plot_matched_factor_dist(matched_factor_dist)
fplot.plot_matched_covariate_dist(matched_covariate_dist, covariate_levels=all_covariate_levels)



### select the factors that are matched with any covariate level
matched_factor_index = np.where(matched_factor_dist>0)[0] 
### subset mean_importance_df to the matched factors
mean_importance_df_matched = mean_importance_df.iloc[:,matched_factor_index] 
## subset x axis labels based on het matched factors
x_labels_matched = mean_importance_df_matched.columns.values

fplot.plot_all_factors_levels_df(mean_importance_df_matched, x_axis_label=x_labels_matched,
                                 title='F-C Match: Feature importance scores', color='coolwarm')

#### calculate the correlation of factors with library size
def get_factor_libsize_correlation(factor_scores, library_size):
    factor_libsize_correlation = np.zeros(factor_scores.shape[1])
    for i in range(factor_scores.shape[1]):
        factor_libsize_correlation[i] = np.corrcoef(factor_scores[:,i], library_size)[0,1]
    return factor_libsize_correlation

library_size = data.obs.nCount_originalexp
factor_libsize_correlation = get_factor_libsize_correlation(factor_scores, library_size)
### create a barplot of the factor_libsize_correlation

def plot_barplot(factor_libsize_correlation, x_labels=None, title=''):
    plt.figure(figsize=(15,5))
    if x_labels is None:
        ### set the x axis labels as F1, F2, ...
        x_labels = ['F'+str(i+1) for i in range(factor_libsize_correlation.shape[0])]
    ### set background color to white
    plt.rcParams['axes.facecolor'] = 'white'
    ## add y and x black axis 
    plt.axhline(y=0, color='black', linewidth=2)
    plt.bar(x_labels, factor_libsize_correlation, color='black')
    ## set y range from 0 to 1
    plt.ylim(-0.5,1)
    plt.xticks(fontsize=25, rotation=90)
    plt.yticks(fontsize=25)
    plt.xlabel('Factors', fontsize=28)
    plt.ylabel('Correlation with library size', fontsize=28)
    plt.title(title, fontsize=26)
    plt.show()

plot_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size')







####################################
#### evaluating bimodality score using simulated factors ####
####################################

#bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
#      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores, time_eff=False)
#bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores, time_eff=False)

silhouette_scores_km, vrs_km = fmet.get_kmeans_scores(factor_scores, time_eff=True)
# silhouette_scores_gmm = fmet.get_gmm_scores(factor_scores, time_eff=True)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
bimodality_scores = bimodality_index_scores
### calculate the average between the silhouette_scores_km, vrs_km and bimodality_index_scores
bimodality_scores = np.mean([silhouette_scores_km, bimodality_index_scores], axis=0)


### label dependent factor metrics
ASV_arith_cell = fmet.get_ASV_all(factor_scores, covariate_vector=y_cell_line, mean_type='arithmetic')
ASV_arith_sample = fmet.get_ASV_all(factor_scores, y_sample, mean_type='arithmetic')

meanimp_simpson = fmet.get_all_factors_simpson_D_index(mean_importance_df)


### calculate diversity metrics
## simpson index: High scores (close to 1) indicate high diversity - meaning that the factor is not specific to any covariate level
## low simpson index (close to 0) indicate low diversity - meaning that the factor is specific to a covariate level
factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df) ## calculated for each factor in the importance matrix

#### label free factor metrics
factor_variance_all = fmet.get_factor_variance_all(factor_scores)


####################################
##### Factor metrics #####
####################################
all_metrics_dict = {'Bimodality':bimodality_scores, 
                    'Specificity':factor_simpson_meanimp,
                    'Effect size': factor_variance_all,
                    'Homogeneity (cell line)':ASV_arith_cell,
                    'Homogeneity (protocol)':ASV_arith_sample}


### check the length of all the metrics

all_metrics_df = pd.DataFrame(all_metrics_dict)
factor_metrics = list(all_metrics_df.columns)
all_metrics_df.head()

all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)

fplot.plot_metric_correlation_clustermap(all_metrics_df)
fplot.plot_metric_dendrogram(all_metrics_df)
fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')

### plot the factors 0:15
fplot.plot_metric_heatmap(all_metrics_scaled[0:15,:], factor_metrics, title='')

###

### subset all_merrics_scaled numpy array to only include the matched factors
all_metrics_scaled_matched = all_metrics_scaled[matched_factor_index,:]
fplot.plot_metric_heatmap(all_metrics_scaled_matched, factor_metrics, x_axis_label=x_labels_matched,
                          title='Scaled metrics for all the factors')

## subset x axis labels based on het matched factors
x_labels_matched = mean_importance_df_matched.columns.values



#####################################################################
### correlation between silhouette_scores_km, vrs_km and bimodality_index_scores
bimodality_corr = np.corrcoef([silhouette_scores_km, vrs_km, bimodality_index_scores])
bimodality_corr_df = pd.DataFrame(bimodality_corr)
bimodality_corr_df.index = ['silhouette_km', 'vrs_km', 'bimodality_index']
bimodality_corr_df.columns = ['silhouette_km', 'vrs_km', 'bimodality_index']
bimodality_corr_df
### calculate ASV based on entropy on the scaled variance per covariate for each factor
ASV_entropy_sample = fmet.get_factor_entropy_all(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_sample)))
ASV_entropy_cell = fmet.get_factor_entropy_all(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_cell_line)))

## calculate correlation between all ASV scores
ASV_list = [ASV_geo_sample, ASV_geo_cell,
            ASV_arith_sample,ASV_arith_cell, 
            meanimp_simpson_sample, meanimp_simpson_cell,
            ASV_simpson_sample, ASV_simpson_cell,
            ASV_entropy_sample, ASV_entropy_cell]
ASV_names = ['ASV_geo_sample', 'ASV_geo_cell',
            'ASV_arith_sample','ASV_arith_cell', 
            'meanimp_simpson_sample', 'meanimp_simpson_cell',
            'ASV_simpson_sample', 'ASV_simpson_cell',
            'ASV_entropy_sample', 'ASV_entropy_cell']


### calculate the correlation between all ASV scores without a function
ASV_corr = np.zeros((len(ASV_list), len(ASV_list)))
for i in range(len(ASV_list)):
    for j in range(len(ASV_list)):
        ASV_corr[i,j] = np.corrcoef(ASV_list[i], ASV_list[j])[0,1]
ASV_corr_df = pd.DataFrame(ASV_corr)
### set the row and column names of ASV_corr_df
ASV_corr_df.index = ASV_names
ASV_corr_df.columns = ASV_names
## make a heatmap of the ASV_corr_df
plt.figure(figsize=(15,12))
plt.imshow(ASV_corr_df, cmap='coolwarm')
plt.xticks(np.arange(ASV_corr_df.shape[1]), ASV_corr_df.columns.values, rotation=90, fontsize=30)
plt.yticks(np.arange(ASV_corr_df.shape[0]), ASV_corr_df.index.values, fontsize=30)
plt.xlabel('ASV scores', fontsize=34)
plt.ylabel('ASV scores', fontsize=34)
plt.title('Correlation between ASV scores', fontsize=34)
plt.show()
## clustermapping the ASV_corr_df
import seaborn as sns
sns.clustermap(ASV_corr_df, cmap='coolwarm', figsize=(15,12), row_cluster=True, col_cluster=True)

plt.show()

