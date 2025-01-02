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
NUM_COMP_TO_VIS = 5


metadata_path = "/home/delaram/sciRED//review_analysis/spatial/spatial_Dorsolateralcortext_metadata.csv"
matrix_path = "/home/delaram//sciRED/review_analysis/spatial/spatial_Dorsolateralcortext_matrix.npz"
row_names_path = "/home/delaram//sciRED/review_analysis/spatial/spatial_Dorsolateralcortext_row_names.csv"
col_names_path = "/home/delaram//sciRED/review_analysis/spatial/spatial_Dorsolateralcortext_col_names.csv"

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
data_all = data[gene_idx,:]



### extract metadata sample_name as sample
y_sample = metadata['sample_name']
y_cluster = metadata['Cluster']
y_spatial_LIBD = metadata['spatialLIBD']
y_subject = metadata['subject']
y_replicate = metadata['replicate']
y_library_size = metadata['sum']
y_cell_id = metadata['Unnamed: 0']
num_genes, num_cells = data.shape

### split the data based on the three subjects
subject_levels = y_subject.unique()

i = 2
a_subject_level = subject_levels[i]
print(a_subject_level)
data = data_all[:, y_subject == a_subject_level]

### subset all y_ variables based on the subject
y_sample = y_sample[y_subject == a_subject_level]
y_cluster = y_cluster[y_subject == a_subject_level]
y_spatial_LIBD = y_spatial_LIBD[y_subject == a_subject_level]
y_replicate = y_replicate[y_subject == a_subject_level]
y_library_size = y_library_size[y_subject == a_subject_level]
y_cell_id = y_cell_id[y_subject == a_subject_level]


gene_sums = np.sum(data,axis=1) # row sums - library size
cell_sums = np.sum(data,axis=0) # col sums - sum reads in a gene

print(data.shape)
print(len(gene_sums))
print(len(cell_sums))

data = data.T #num cells, num genes

#### design matrix - library size only
x = y_library_size
## add a vector of ones to the design matrix (before library size)
x = np.column_stack((np.ones(x.shape[0]), x))


########################################
#### design matrix - library size and sample
column_levels = y_sample.unique() 
dict_covariate = {}
for column_level in column_levels:
    dict_covariate[column_level] = proc.get_binary_covariate(y_sample.squeeze(), column_level)
#### stack colummns of dict_covariate 
x_sample = np.column_stack(([dict_covariate[column] for column in column_levels]))

x = np.column_stack((y_library_size, x_sample)) 
x = sm.add_constant(x) ## adding the intercept
### print head of x
print(x[:5,:])


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
pipeline = Pipeline([('scaling', StandardScaler()), 
                     ('pca', PCA(n_components=NUM_COMPONENTS))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_ 
pca_loading.shape #(factors, genes)

plt.plot(pca.explained_variance_ratio_)


my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
            for i in np.unique(y_replicate)}
replicate_color = [my_color[y_replicate.iloc[i]] for i in range(len(y_replicate))]

my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
            for i in np.unique(y_sample)}
sample_color = [my_color[y_sample.iloc[i]] for i in range(len(y_sample))]

my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for i in np.unique(y_subject)}
subject_color = [my_color[y_subject.iloc[i]] for i in range(len(y_subject))]

# Convert all values in y_spatial_LIBD to strings
y_spatial_LIBD = y_spatial_LIBD.astype(str)
my_color = {i: "#"+''.join([random.choice('1023456789ABCDEF') for j in range(6)])
            for i in np.unique(y_spatial_LIBD)}
spatial_LIBD_color = [my_color[y_spatial_LIBD.iloc[i]] for i in range(len(y_spatial_LIBD))]

#plt_legend_sample = exvis.get_legend_patch(y_sample, colors_dict_humanPBMC['sample'] )

### make a dictionary of colors for each sample in y_sample
vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
             cell_color_vec= replicate_color,
               legend_handles=False,
               title='PCA of dorso lateral cortex data - replicate')

vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
             cell_color_vec= sample_color,
               legend_handles=False,
               title='PCA of dorso lateral cortex data - sample')

vis.plot_pca(pca_scores, NUM_COMP_TO_VIS, 
             cell_color_vec= spatial_LIBD_color,
               legend_handles=False,
               title='PCA of dorso lateral cortex data - spatial_LIBD')

vis.plot_pca(pca_scores, NUM_COMP_TO_VIS,
              cell_color_vec= subject_color,
                legend_handles=False,
                title='PCA of dorso lateral cortex data - subject')
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
NUM_COMP_TO_VIS = 7
vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS, 
             cell_color_vec= replicate_color,
               legend_handles=False,
               title='varimax-PCA of pearson residuals\n dorso lateral cortex data - replicate')

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS,
              cell_color_vec= sample_color,
                legend_handles=False,
                title='varimax-PCA of pearson residuals\n dorso lateral cortex data - sample')

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS,
              cell_color_vec= spatial_LIBD_color,
                legend_handles=False,
                title='varimax-PCA of pearson residuals\n dorso lateral cortex data - spatial_LIBD')

vis.plot_pca(pca_scores_varimax, NUM_COMP_TO_VIS,
              cell_color_vec= subject_color,
                legend_handles=False,
                title='varimax-PCA of pearson residuals\n dorso lateral cortex data - subject')

varimax_loading_df = pd.DataFrame(varimax_loading)
varimax_loading_df.columns = ['F'+str(i) for i in range(1, varimax_loading_df.shape[1]+1)]
varimax_loading_df.index = genes[gene_idx]


### save the varimax_loading_df and varimax_scores to a csv file
pca_scores_varimax_df = pd.DataFrame(pca_scores_varimax)
pca_scores_varimax_df.columns = ['F'+str(i) for i in range(1, pca_scores_varimax_df.shape[1]+1)]

### subset cells pandas.core.series.Series based on subject
pca_scores_varimax_df.index = y_cell_id
### save the varimax_loading_df and varimax_scores to a csv file
#varimax_loading_df.to_csv('/home/delaram/sciRED/review_analysis/spatial/varimax_loading_df_spatial_Dorsolateralcortext_'+a_subject_level+'.csv')
#pca_scores_varimax_df.to_csv('/home/delaram/sciRED/review_analysis/spatial/pca_scores_varimax_df_spatial_Dorsolateralcortext_'+a_subject_level+'.csv')

varimax_loading_df.to_csv('/home/delaram/sciRED/review_analysis/spatial/varimax_loading_df_spatial_Dorsolateralcortext_'+a_subject_level+'_reg_libsubj.csv')
pca_scores_varimax_df.to_csv('/home/delaram/sciRED/review_analysis/spatial/pca_scores_varimax_df_spatial_Dorsolateralcortext_'+a_subject_level+'_reg_libsubj.csv')


### read the varimax_loading_df and varimax_scores from a csv file
#varimax_loading_df = pd.read_csv('/home/delaram/sciRED/review_analysis/spatial/varimax_loading_df_spatial_Dorsolateralcortext_'+a_subject_level+'.csv', index_col=0)
#pca_scores_varimax_df = pd.read_csv('/home/delaram/sciRED/review_analysis/spatial/pca_scores_varimax_df_spatial_Dorsolateralcortext_'+a_subject_level+'.csv', index_col=0)
#varimax_loading = varimax_loading_df.values
#pca_scores_varimax = pca_scores_varimax_df.values

########################
######## PCA factors
factor_loading = pca_loading
factor_scores = pca_scores

##### Varimax factors
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax
covariate_vec = y_cluster
covariate_level = np.unique(covariate_vec)[1]

####################################
#### FCAT score calculation ######
####################################

### FCAT needs to be calculated for each covariate separately
fcat_spatial_LIBD = efca.FCAT(y_spatial_LIBD, factor_scores, scale='standard', mean='arithmatic')
fcat_sample = efca.FCAT(y_sample, factor_scores, scale='standard', mean='arithmatic')
fcat_replicate = efca.FCAT(y_replicate, factor_scores, scale='standard', mean='arithmatic')


### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_spatial_LIBD, fcat_sample, fcat_replicate], axis=0)
### remove the rownames called NA or nan from table
fcat = fcat[fcat.index != 'NA']
fcat = fcat[fcat.index != 'nan']
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
                    save=False, save_path='../Plots/mean_importance_df_matched_spatial_Dorsolateralcortext.pdf')

factor_libsize_correlation = corr.get_factor_libsize_correlation(factor_scores, library_size = y_library_size)
vis.plot_factor_cor_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size', 
             y_label='Correlation', x_label='Factors')


### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_spatial_LIBD], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table
fcat = fcat[fcat.index != 'nan'] ### remove the rownames called nan from table

vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
              x_axis_tick_fontsize=36, y_axis_tick_fontsize=40, 
              save=False, save_path='../Plots/mean_importance_df_matched_spatial_Dorsolateralcortext.pdf')


#cluster_fcat_sorted_scores, cluster_factors_sorted = vis.plot_sorted_factor_FCA_scores(fcat, 'cluster')

### select the factors that are matched with any covariate level
matched_factor_index = np.where(matched_factor_dist>0)[0] 
fcat_matched = fcat.iloc[:,matched_factor_index] 
x_labels_matched = fcat_matched.columns.values
vis.plot_FCAT(fcat_matched, x_axis_label=x_labels_matched, title='', color='coolwarm',
              x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
              x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, 
              save=False, save_path='../Plots/mean_importance_df_matched_spatial_Dorsolateralcortext.pdf')


####################################
#### Bimodality scores
bimodality_index = met.bimodality_index(factor_scores)
bimodality_score = bimodality_index
#### Effect size
factor_variance = met.factor_variance(factor_scores)

## Specificity
simpson_fcat = met.simpson_diversity_index(fcat)

### label dependent factor metrics
asv_spatial_LIBD = met.average_scaled_var(factor_scores, covariate_vector=y_spatial_LIBD, mean_type='arithmetic')
asv_subject = met.average_scaled_var(factor_scores, covariate_vector=y_subject, mean_type='arithmetic')
asv_sample = met.average_scaled_var(factor_scores, y_sample, mean_type='arithmetic')
asv_replicate = met.average_scaled_var(factor_scores, y_replicate, mean_type='arithmetic')


########### create factor-interpretibility score table (FIST) ######
metrics_dict = {'Bimodality':bimodality_score, 
                    'Specificity':simpson_fcat,
                    'Effect size': factor_variance,
                    'Homogeneity (spatial LIBD)':asv_spatial_LIBD,
                    'Homogeneity (subject)':asv_subject,
                    'Homogeneity (replicate)':asv_replicate,
                    'Homogeneity (sample)':asv_sample}

fist = met.FIST(metrics_dict)
vis.plot_FIST(fist, title='Scaled metrics for all the factors')
### subset the first 15 factors of fist dataframe
vis.plot_FIST(fist.iloc[0:15,:])
vis.plot_FIST(fist.iloc[matched_factor_index,:])





