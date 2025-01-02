## sparsity analysis on human liver data
# Specific analysis and Metrics to include: 
#Average variance explained for all factors for each iteration of factor analysis (how rotation is affecting this) - should go down with sparsity
#Look into percentage matched factors for each dataset - would be nice if doesn’t change much


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


def get_sparse_data(y, sparsity=0.05):
    y_sparse = y.copy()
    num_zeros = int(y_sparse.size * sparsity)
    idx = np.random.choice(y_sparse.size, num_zeros, replace=False)
    y_sparse.ravel()[idx] = 0
    return y_sparse

def get_variance_explained(scores):
    return np.var(scores, axis=0) / np.sum(np.var(scores, axis=0))

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

### fit GLM to each gene
glm_fit_dict = glm.poissonGLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
y = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y.shape) # (num_cells, num_genes)



### write a look and randomly replace 5,10,20,50% of the data with 0 and see how it affects the factor analysis

## make a dictionary to store the results for pca variance explained for each sparsity level
pca_variance_explained = {}
varimax_variance_explained = {}
percent_matched_fact_dict = {}
percent_matched_cov_dict = {}

### count the number of zeros in the data
num_zeros = np.sum(y == 0)
print('num_zeros: ', num_zeros, 'total: ', y.size, 'percent: ', num_zeros/y.size)



for sparsity in [0.01, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:

    #### enforcing sparsity to the normalized data
    #y_sparse = get_sparse_data(y, sparsity=sparsity)

    ## enforce sparsity to the count data
    y_copy = y.copy()
    y_sparse = get_sparse_data(y_copy, sparsity=sparsity)

    ## as a sanity check, aount the number of zeros and devide by total 
    num_zeros = np.sum(y_sparse == 0)
    print('sparsity: ', sparsity, 'num_zeros: ', num_zeros, 'total: ', 
          y_sparse.size, 'percent: ', num_zeros/y_sparse.size)

    ### fit GLM to each gene
    glm_fit_dict = glm.poissonGLM(y_sparse, x)
    resid_pearson = glm_fit_dict['resid_pearson'] 
    print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
    print('y shape: ', y_sparse.shape) # (num_cells, num_genes)
    y_sparse = resid_pearson.T # (num_cells, num_genes)
    print('y shape: ', y_sparse.shape) # (num_cells, num_genes)

    pipeline = Pipeline([('scaling', StandardScaler()), 
                         ('pca', PCA(n_components=NUM_COMPONENTS))])
    pca_scores = pipeline.fit_transform(y_sparse)
    pca = pipeline.named_steps['pca']
    pca_loading = pca.components_
    pca_loading.shape
    plt.plot(pca.explained_variance_ratio_)
    pca_variance_explained[sparsity] = pca.explained_variance_ratio_

    ######## Applying varimax rotation to the factor scores
    rotation_results_varimax = rot.varimax(pca_loading.T)
    varimax_loading = rotation_results_varimax['rotloading']
    pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

    varimax_variance_explained[sparsity] = get_variance_explained(pca_scores_varimax)

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
    matched_factor_dist, percent_matched_fact = efca.get_percent_matched_factors(fcat,
                                                                                    threshold)
    matched_covariate_dist, percent_matched_cov = efca.get_percent_matched_covariates(fcat,
                                                                                        threshold=threshold)
    print('percent_matched_fact: ', percent_matched_fact)
    print('percent_matched_cov: ', percent_matched_cov)
    percent_matched_fact_dict[sparsity] = percent_matched_fact
    percent_matched_cov_dict[sparsity] = percent_matched_cov
    vis.plot_matched_factor_dist(matched_factor_dist)
    vis.plot_matched_covariate_dist(matched_covariate_dist,
                                    covariate_levels=all_covariate_levels)
    


### calculate average explain variance for all factors in varimax_variance_explained
average_variance_explained_varimax = {}
average_variance_explained_pca = {}
for sparsity in varimax_variance_explained.keys():
    average_variance_explained_varimax[sparsity] = np.mean(varimax_variance_explained[sparsity])
    average_variance_explained_pca[sparsity] = np.mean(pca_variance_explained[sparsity])

### calculate geometric mean (instead of arithmatic) of the average variance explained for all factors

geo_average_variance_explained_varimax = {}
geo_average_variance_explained_pca = {}
for sparsity in varimax_variance_explained.keys():
    geo_average_variance_explained_varimax[sparsity] = np.prod(varimax_variance_explained[sparsity])**(1/NUM_COMPONENTS)
    geo_average_variance_explained_pca[sparsity] = np.prod(pca_variance_explained[sparsity])**(1/NUM_COMPONENTS)

# max variance explained for each sparsity level
max_variance_explained_varimax = {}
max_variance_explained_pca = {}
for sparsity in varimax_variance_explained.keys():
    max_variance_explained_varimax[sparsity] = np.max(varimax_variance_explained[sparsity])
    max_variance_explained_pca[sparsity] = np.max(pca_variance_explained[sparsity])


# plot pca and varimax average_variance_explained on y and sparsity level in x - write from scratch
plt.figure(figsize=(10, 6))
plt.plot(list(average_variance_explained_varimax.keys()), 
         list(average_variance_explained_varimax.values()),
            label='varimax',linewidth=2, 
            marker='o')
plt.plot(list(average_variance_explained_pca.keys()), 
         list(average_variance_explained_pca.values()),
            label='pca', linewidth=2, marker='o')
plt.xlabel('Sparsity level', fontsize=18)
plt.ylabel('Average variance explained', fontsize=18)
plt.legend(fontsize=16)
## increase tick size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Average variance explained for all factors', fontsize=18)
plt.show()

#### create the same plot for geometric mean
plt.figure(figsize=(10, 6))
plt.plot(list(geo_average_variance_explained_varimax.keys()), 
         list(geo_average_variance_explained_varimax.values()),
            label='varimax', linewidth=2, marker='o')
plt.plot(list(geo_average_variance_explained_pca.keys()),
            list(geo_average_variance_explained_pca.values()),
                label='pca', linewidth=2, marker='o')
plt.xlabel('Sparsity level', fontsize=18)
plt.ylabel('Geometric mean variance explained', fontsize=18)
plt.legend(fontsize=16)
## increase tick size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Geometric mean variance explained for all factors', fontsize=18)
plt.show()



#### create the same plot for max variance explained
plt.figure(figsize=(10, 6))
plt.plot(list(max_variance_explained_varimax.keys()), 
         list(max_variance_explained_varimax.values()),
            label='varimax', linewidth=2, marker='o')
plt.plot(list(max_variance_explained_pca.keys()),
            list(max_variance_explained_pca.values()),
                label='pca',linewidth=2,  marker='o')
plt.xlabel('Sparsity level', fontsize=18)
plt.ylabel('Max variance explained', fontsize=18)
plt.legend(fontsize=16)
## increase tick size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Max variance explained for all factors', fontsize=18)
plt.show()


### sort percent_matched_fact_dict by key values
percent_matched_fact_dict = dict(sorted(percent_matched_fact_dict.items()))

# plot percent_matched_fact_dict
plt.figure(figsize=(10, 6))
plt.plot(list(percent_matched_fact_dict.keys()), 
         list(percent_matched_fact_dict.values()),
            label='percent_matched_fact', 
            ### increase line width and change color to purple
            linewidth=2, color='purple',
            marker='o')
## start the y axis from 0
plt.ylim(0, 100)
plt.xlabel('Sparsity level', fontsize=18)
plt.ylabel('Percent matched factors', fontsize=18)
plt.legend(fontsize=16)
## increase tick size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Percent matched factors', fontsize=20)
plt.show()


### sort percent_matched_cov_dict by key values
percent_matched_cov_dict = dict(sorted(percent_matched_cov_dict.items()))
# plot percent_matched_cov_dict
plt.figure(figsize=(10, 6))
plt.plot(list(percent_matched_cov_dict.keys()), 
         list(percent_matched_cov_dict.values()),
            label='percent_matched_cov', 
            ### increase line width and change color to purple
            linewidth=2, color='purple',
            marker='o') 
plt.xlabel('Sparsity level', fontsize=18)
plt.ylabel('Percent matched covariates', fontsize=18)
plt.legend(fontsize=16)
plt.ylim(0, 110)

## increase tick size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Percent matched covariates', fontsize=20)
plt.show()
