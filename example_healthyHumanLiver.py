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
import umap

np.random.seed(10)
NUM_COMPONENTS = 30
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5

data_file_path =  '/home/delaram/sciFA/Data/HumanLiverAtlas.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs


y, genes, num_cells, num_genes = proc.get_data_array(data)

#pd.DataFrame(genes).to_csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv', index=False)
y_sample, y_cell_type = exproc.get_metadata_humanLiver(data)
colors_dict_humanLiver = exvis.get_colors_dict_humanLiver(y_sample, y_cell_type)
plt_legend_sample = exvis.get_legend_patch(y_sample, colors_dict_humanLiver['sample'] )
plt_legend_cell_type = exvis.get_legend_patch(y_cell_type, colors_dict_humanLiver['cell_type'] )


#### design matrix - library size only
x = proc.get_library_design_mat(data, lib_size='total_counts')

#### design matrix - library size and sample
#x_sample = proc.get_design_mat(metadata_col='sample', data=data) 
#x = np.column_stack((data.obs.total_counts, x_sample)) 
#x = sm.add_constant(x) ## adding the intercept

### fit GLM to each gene
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
pipeline = Pipeline([('scaling', StandardScaler()), 
                     ('pca', PCA(n_components=NUM_COMPONENTS))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape
plt.plot(pca.explained_variance_ratio_)


title = 'PCA of pearson residuals - lib size/protocol removed'
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_humanLiver['cell_type'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_type)

vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_humanLiver['sample'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_sample)


#### plot the loadings of the factors
vis.plot_factor_loading(pca_loading.T, genes, 0, 1, fontsize=10, 
                    num_gene_labels=2,title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)


vis.plot_umap(pca_scores, 
              title='UMAP',
              cell_color_vec= colors_dict_humanLiver['cell_type'] , 
               legend_handles=True,plt_legend_list=plt_legend_cell_type)

vis.plot_umap(pca_scores, 
              title='UMAP',
              cell_color_vec= colors_dict_humanLiver['sample'] , 
               legend_handles=True,plt_legend_list=plt_legend_sample)


######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

title = 'Varimax PCA of pearson residuals'
vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_humanLiver['cell_type'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_type)

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_humanLiver['sample'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_sample)

#### plot the loadings of the factors
vis.plot_factor_loading(varimax_loading, genes, 0, 4, fontsize=10, 
                    num_gene_labels=6,title='Scatter plot of the loading vectors', 
                    label_x=False, label_y=False)


########################
######## PCA factors
factor_loading = pca_loading
factor_scores = pca_scores

##### Varimax factors
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax
covariate_vec = y_cell_type
covariate_level = np.unique(covariate_vec)[1]


####################################
#### FCAT score calculation ######
####################################

### FCAT needs to be calculated for each covariate separately
fcat_sample = efca.FCAT(y_sample, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_type = efca.FCAT(y_cell_type, factor_scores, scale='standard', mean='arithmatic')


### plot the FCAT scores
fcat = pd.concat([fcat_sample, fcat_cell_type], axis=0)
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
## add index 19 to the matched_factor_index
matched_factor_index = np.append(matched_factor_index, 19)
fcat_matched = fcat.iloc[:,matched_factor_index] 
x_labels_matched = fcat_matched.columns.values
vis.plot_FCAT(fcat_matched, x_axis_label=x_labels_matched, title='', color='coolwarm',
                                 x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                                 x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, 
                                 save=False, save_path='../Plots/mean_importance_df_matched_ratliver.pdf')



factor_libsize_correlation = corr.get_factor_libsize_correlation(factor_scores, 
                                                                 library_size = data.obs.total_counts)
vis.plot_factor_cor_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size', 
             y_label='Correlation', x_label='Factors')


#### concatenate the factor scores with the metadata and umap embedding
reducer = umap.UMAP()
embedding = reducer.fit_transform(factor_scores)
factor_scores_df = pd.DataFrame(factor_scores)
factor_scores_df.columns = ['factor_{}'.format(i) for i in range(factor_scores.shape[1])]
factor_scores_df['SAMPLE'] = y_sample
factor_scores_df['CELL_TYPE'] = y_cell_type
factor_scores_df['umap_1'] = embedding[:,0]
factor_scores_df['umap_2'] = embedding[:,1]

for col in data.obs.columns:
    factor_scores_df[col] = data.obs[col].values
### add rownames of data.obs to the factor_scores_df
factor_scores_df['id'] = data.obs.index.values
#factor_scores_df.to_csv('/home/delaram/sciFA/Results/factor_scores_umap_df_humanlivermap.csv', index=False)
#pd.DataFrame(factor_loading).to_csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv', index=False)



####################################
#### Bimodality scores
silhouette_score = met.kmeans_bimodal_score(factor_scores, time_eff=True)
bimodality_index = met.bimodality_index(factor_scores)
bimodality_score = np.mean([silhouette_score, bimodality_index], axis=0)
bimodality_score = bimodality_index
#### Effect size
factor_variance = met.factor_variance(factor_scores)

## Specificity
simpson_fcat = met.simpson_diversity_index(fcat)

### label dependent factor metrics
asv_cell_type = met.average_scaled_var(factor_scores, covariate_vector=y_cell_type, mean_type='arithmetic')
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
### include factors F10, F19, F26, F28, F30
vis.plot_FIST(fist.iloc[[9, 18, 25, 27, 29],:], 
              x_axis_label=['F10', 'F19', 'F26', 'F28', 'F30'])
vis.plot_FIST(fist.iloc[matched_factor_index,:])



################################################################
########  Creating the FIS table for a subset of factors ########
################################################################
#### Bimodality scores
### subset factor scores to include factors F10, F19, F26, F28, F30
selected_factors = [9, 18, 25, 27, 29]
factor_scores_subset = factor_scores[:,selected_factors]
silhouette_score = met.kmeans_bimodal_score(factor_scores, time_eff=True)
bimodality_index = met.bimodality_index(factor_scores)
bimodality_score = np.mean([silhouette_score, bimodality_index], axis=0)
bimodality_score = bimodality_index
#### Effect size
factor_variance = met.factor_variance(factor_scores)

## Specificity
simpson_fcat = met.simpson_diversity_index(fcat)

### label dependent factor metrics
asv_cell_type = met.average_scaled_var(factor_scores, covariate_vector=y_cell_type, mean_type='arithmetic')
asv_sample = met.average_scaled_var(factor_scores, y_sample, mean_type='arithmetic')


########### create factor-interpretibility score table (FIST) ######
metrics_dict = {'Bimodality':bimodality_score, 
                    'Specificity':simpson_fcat,
                    'Effect size': factor_variance,
                    'Homogeneity (cell type)':asv_cell_type,
                    'Homogeneity (sample)':asv_sample}

fist = met.FIST(metrics_dict)