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
import random

NUM_GENES = 2000
NUM_COMPONENTS = 30
NUM_COMP_TO_VIS = 5

data_file_path = '/home/delaram/sciRED/review_analysis/Ischemia_Reperfusion_Responses_Human_Lung_Transplants.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)

### make a table of all the metadata columns (data.obs)
metadata = data.obs
print(metadata.head())
print(metadata.columns)
print(metadata.describe())
print(metadata.describe(exclude=[np.number]))

y_lib_size = metadata[['nCount_RNA']].squeeze()
y_timepoint = metadata[['timepoint']].squeeze()
y_phase = metadata[['Phase']].squeeze()

y_case = metadata[['ltx_case']].squeeze()
y_sample = metadata[['sample_name']].squeeze()
y_donor_id = metadata[['donor_id']].squeeze()
y_recipient_origin = metadata[['recipient_origin']].squeeze()
y_cell_type = metadata[['cell_type']].squeeze()
y_sex = metadata[['sex']].squeeze()
y_tissue = metadata[['tissue']].squeeze()

### save metadata as a csv file
#metadata.to_csv('/home/delaram/sciRED/review_analysis/Ischemia_Reperfusion_Responses_Human_Lung_Transplants_metadata.csv', index=False)

### prnt the unique values of each of the vectors defined above
print(y_timepoint.unique())
print(y_phase.unique())
print(y_case.unique())
print(y_sample.unique())
print(y_recipient_origin.unique())
print(y_cell_type.unique())
print(y_sex.unique())
print(y_tissue.unique())


### check if data is normalized or not
print('data is normalized: ', np.allclose(np.sum(y, axis=1), 1))
y, genes, num_cells, num_genes = proc.get_data_array(data)

####################################
#### fit GLM to each gene ######
####################################

#### design matrix - library size only
x = y_lib_size
## adding the intercept
x = sm.add_constant(x) ## adding the intercept


#### design matrix - library size and sample
column_levels = y_case.unique() 
dict_covariate = {}
for column_level in column_levels:
    dict_covariate[column_level] = proc.get_binary_covariate(y_case.squeeze(), column_level)
#### stack colummns of dict_covariate 
x_sample = np.column_stack(([dict_covariate[column] for column in column_levels]))

x = np.column_stack((y_lib_size, x_sample)) 
x = sm.add_constant(x) ## adding the intercept


### fit GLM to each gene
glm_fit_dict = glm.poissonGLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
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
            for i in np.unique(y_cell_type)}
cell_type_color = [my_color[y_cell_type.iloc[i]] for i in range(len(y_cell_type))]


my_color = {i: "#"+''.join([random.choice('1023456789ABCDEF') for j in range(6)]) 
            for i in np.unique(y_case)}
case_color = [my_color[y_case.iloc[i]] for i in range(len(y_case))]



### make a dictionary of colors for each sample in y_sample
vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
             cell_color_vec= cell_type_color,
               legend_handles=False,
               title='PCA')

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
             cell_color_vec= cell_type_color,
               legend_handles=False,
               title='varimax-PCA of pearson residuals\n lung- cell type')

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS,
              cell_color_vec= case_color,
                legend_handles=False,
                title='varimax-PCA of pearson residuals\n lung - sample')

varimax_loading_df = pd.DataFrame(varimax_loading)
varimax_loading_df.columns = ['F'+str(i) for i in range(1, varimax_loading_df.shape[1]+1)]
## add gene names to the varimax_loading_df
varimax_loading_df['gene'] = genes
### save the varimax_loading_df to a csv file
#varimax_loading_df.to_csv('/home/delaram/sciRED/review_analysis/varimax_loading_lung.csv', index=False)

### save the varimax_loading_df and varimax_scores to a csv file
pca_scores_varimax_df = pd.DataFrame(pca_scores_varimax)
pca_scores_varimax_df.columns = ['F'+str(i) for i in range(1, pca_scores_varimax_df.shape[1]+1)]
### add all metadata columns to the pca_scores_varimax_df

## merge metadata to pca_scores_varimax_df so the metadata columns are added to the pca_scores_varimax_df
## add metadata index to it as a column called cell_id
metadata = data.obs
metadata['cell_id'] = metadata.index
metadata = metadata.reset_index(drop=True)
pca_scores_varimax_df['cell_id'] = metadata['cell_id']
pca_scores_varimax_df = pd.merge(pca_scores_varimax_df, metadata, on='cell_id', how='inner')
pca_scores_varimax_df.head()
print(pca_scores_varimax_df.shape)
#pca_scores_varimax_df.to_csv('/home/delaram/sciRED/review_analysis/varimax_scores_lung.csv', index=False)

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
fcat_case = efca.FCAT(y_case, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_type = efca.FCAT(y_cell_type, factor_scores, scale='standard', mean='arithmatic')
fcat_recipient = efca.FCAT(y_recipient_origin, factor_scores, scale='standard', mean='arithmatic')
fcat_timepoint = efca.FCAT(y_timepoint, factor_scores, scale='standard', mean='arithmatic')
fcat_tissue = efca.FCAT(y_tissue, factor_scores, scale='standard', mean='arithmatic')
fcat_sex = efca.FCAT(y_sex, factor_scores, scale='standard', mean='arithmatic')



### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_cell_type, fcat_recipient, fcat_timepoint, 
                  fcat_tissue, fcat_sex, fcat_case], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table

vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)


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
                                 save=False, save_path='../Plots/mean_importance_df_matched_lung.pdf')

factor_libsize_correlation = corr.get_factor_libsize_correlation(factor_scores, library_size = y_lib_size)
vis.plot_factor_cor_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size', 
             y_label='Correlation', x_label='Factors')


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

bimodality_index = met.bimodality_index(factor_scores)
bimodality_score = bimodality_index
#### Effect size
factor_variance = met.factor_variance(factor_scores)
## Specificity
simpson_fcat = met.simpson_diversity_index(fcat)


### label dependent factor metrics
asv_cell_type = met.average_scaled_var(factor_scores, covariate_vector=y_cell_type, mean_type='arithmetic')
asv_case = met.average_scaled_var(factor_scores, y_case, mean_type='arithmetic')
asv_donor_id = met.average_scaled_var(factor_scores, y_donor_id, mean_type='arithmetic')
asv_recipient_origin = met.average_scaled_var(factor_scores, y_recipient_origin, mean_type='arithmetic')
asv_sex = met.average_scaled_var(factor_scores, y_sex, mean_type='arithmetic')
asv_tissue = met.average_scaled_var(factor_scores, y_tissue, mean_type='arithmetic')
asv_timepoint = met.average_scaled_var(factor_scores, y_timepoint, mean_type='arithmetic')



########### create factor-interpretibility score table (FIST) ######
metrics_dict = {'Bimodality':bimodality_score, 
                    'Specificity':simpson_fcat,
                    'Effect size': factor_variance,
                    'Homogeneity (cell type)':asv_cell_type,
                    'Homogeneity (case)':asv_case,
                    'Homogeneity (donor_id)':asv_donor_id,
                    'Homogeneity (recipient_origin)':asv_recipient_origin,
                    'Homogeneity (sex)':asv_sex,
                    'Homogeneity (tissue)':asv_tissue,
                    'Homogeneity (timepoint)':asv_timepoint}

fist = met.FIST(metrics_dict)
vis.plot_FIST(fist, title='Scaled metrics for all the factors')
### subset the first 15 factors of fist dataframe
vis.plot_FIST(fist.iloc[0:15,:])
vis.plot_FIST(fist.iloc[matched_factor_index,:])


