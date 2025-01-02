import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
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
NUM_COMPONENTS = 30
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


####################################
#### fit GLM to each gene ######
####################################

#### normalizing the data by regressing library size only
x = proc.get_library_design_mat(data, lib_size='nCount_originalexp')
glm_fit_dict = glm.poissonGLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
y_norm = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y_norm.shape) # (num_cells, num_genes)

###############################################
#### Running PCA on the pearson residual ######
###############################################

### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), 
                     ('pca', PCA(n_components=NUM_COMPONENTS))])
pca_scores = pipeline.fit_transform(y_norm)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape
plt.plot(pca.explained_variance_ratio_)

#### Experiment-1: PCA factors
factor_loading = pca_loading
factor_scores = pca_scores

### FCAT needs to be calculated for each covariate separately
fcat_protocol = efca.FCAT(y_protocol, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line, factor_scores, scale='standard', mean='arithmatic')

### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

title = 'PCA of lib size normalized data'
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(factor_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)

### make a dictionary of colors for each sample in y_sample
vis.plot_pca(factor_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_protocol)



pca_loading_df = pd.DataFrame(pca_loading.T)
pca_loading_df.columns = ['F'+str(i) for i in range(1, pca_loading_df.shape[1]+1)]
### add genes as a column
pca_loading_df.index = genes


### save the pca_loading_df and pca_scores to a csv file
metadata = data.obs
pca_scores_df = pd.DataFrame(pca_scores)
pca_scores_df.columns = ['F'+str(i) for i in range(1, pca_scores_df.shape[1]+1)]
if pca_scores_df.shape[0] == metadata.shape[0]:
    pca_scores_df_merged = pd.concat([metadata.reset_index(drop=True), 
                                      pca_scores_df.reset_index(drop=True)], axis=1)
else:
    raise ValueError("Number of rows in 'metadata' does not match 'pca_scores_df'.")
pca_scores_df_merged.head()


pca_loading_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/pca_loading_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')
pca_scores_df_merged.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/pca_scores_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')



###############################################
#### Running NMF on the count data ######
###############################################
### Apply NMF to the data using scikit-learn without pipeline
start = time.time()
nmf = NMF(n_components=NUM_COMPONENTS, init='nndsvd', random_state=0)
nmf_scores = nmf.fit_transform(y)
nmf_loading = nmf.components_
nmf_loading.shape
end = time.time()

print('Time to fit NMF - numcomp '+ str(NUM_COMPONENTS)+ ' : ', end-start)
print('Time to fit NMF (min) - numcomp '+ str(NUM_COMPONENTS)+ ' : ', (end-start)/60)

factor_loading = nmf_loading
factor_scores = nmf_scores

### FCAT needs to be calculated for each covariate separately
fcat_protocol = efca.FCAT(y_protocol, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line, factor_scores, scale='standard', mean='arithmatic')

### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

title = 'NMF on count data'
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(factor_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)

### make a dictionary of colors for each sample in y_sample
vis.plot_pca(factor_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_protocol)
####################################

nmf_loading_df = pd.DataFrame(nmf_loading)
nmf_loading_df = nmf_loading_df.T
nmf_loading_df.columns = ['F'+str(i) for i in range(1, nmf_loading_df.shape[1]+1)]
nmf_loading_df.index = genes


### save to csv file
nmf_scores_df = pd.DataFrame(nmf_scores)
nmf_scores_df.columns = ['F'+str(i) for i in range(1, nmf_scores_df.shape[1]+1)]
nmf_scores_df.index = data.obs.index.values
nmf_scores_df_merged = pd.concat([data.obs, nmf_scores_df], axis=1)

## add num_components to the file name
nmf_loading_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/nmf_loading_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')
nmf_scores_df_merged.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/nmf_scores_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')


###############################################
#### Running ICA on the pearson residual ######
###############################################
start = time.time()
ica = FastICA(n_components=NUM_COMPONENTS, random_state=0)
ica_scores = ica.fit_transform(y_norm)
ica_loading = ica.mixing_
ica_loading.shape

end = time.time()

print('Time to fit ica - numcomp '+ str(NUM_COMPONENTS)+ ' : ', end-start)
print('Time to fit ica (min) - numcomp '+ str(NUM_COMPONENTS)+ ' : ', (end-start)/60)

factor_loading = ica_loading
factor_scores = ica_scores

### FCAT needs to be calculated for each covariate separately
fcat_protocol = efca.FCAT(y_protocol, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line, factor_scores, scale='standard', mean='arithmatic')
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

title = 'ICA on pearson residuals'
NUM_COMP_TO_VIS = 3
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(factor_scores, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)
vis.plot_pca(factor_scores, NUM_COMP_TO_VIS,
                cell_color_vec= colors_dict_scMix['protocol'], 
                legend_handles=True,
                title=title,
                plt_legend_list=plt_legend_protocol)


ica_loading_df = pd.DataFrame(ica_loading)
ica_loading_df.columns = ['F'+str(i) for i in range(1, ica_loading_df.shape[1]+1)]
### add genes as a column
ica_loading_df.index = genes

### save the ica_loading_df and ica_scores to a csv file
metadata = data.obs
ica_scores_df = pd.DataFrame(ica_scores)
ica_scores_df.columns = ['F'+str(i) for i in range(1, ica_scores_df.shape[1]+1)]
if ica_scores_df.shape[0] == metadata.shape[0]:
    ica_scores_df_merged = pd.concat([metadata.reset_index(drop=True), 
                                      ica_scores_df.reset_index(drop=True)], axis=1)
else:
    raise ValueError("Number of rows in 'metadata' does not match 'ica_scores_df'.")
ica_scores_df_merged.head()

ica_loading_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/ica_loading_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')
ica_scores_df_merged.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/ica_scores_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv')

###############################################

### read scCOGAPs results for scMix data: 
file_name = '/home/delaram/sciRED//review_analysis/benchmark_methods/scCoGAPS_scores_scMix.csv'
scCoGAPS_scores = pd.read_csv(file_name, index_col=0)
scCoGAPS_scores = scCoGAPS_scores.T
scCoGAPS_scores.index = data.obs.index.values
scCoGAPS_scores.head()
### check if cell_line column in scCoGAPS_scores and metadata are the same
print(np.all(data.obs['cell_line'] == scCoGAPS_scores['cell_line']))
y_protocol_scCoGAPS = scCoGAPS_scores['sample']
y_cell_line_scCoGAPS = scCoGAPS_scores['cell_line']

### extract columns with names as Pattern_* from scCoGAPS_scores
pattern_cols = [col for col in scCoGAPS_scores.columns if 'Pattern' in col]
scCoGAPS_scores_factor = scCoGAPS_scores[pattern_cols]
## convert to numpy.ndarray
scCoGAPS_scores_factor = scCoGAPS_scores_factor.to_numpy()


### run FCAT on scCoGAPS results
fcat_protocol = efca.FCAT(y_protocol_scCoGAPS, scCoGAPS_scores_factor, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line_scCoGAPS, scCoGAPS_scores_factor, scale='standard', mean='arithmatic')
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)


vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

title = 'scCoGAPS on scMix data'
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(scCoGAPS_scores_factor, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)
vis.plot_pca(scCoGAPS_scores_factor, NUM_COMP_TO_VIS,
                cell_color_vec= colors_dict_scMix['protocol'], 
                legend_handles=True,
                title=title,
                plt_legend_list=plt_legend_protocol)





### read zinbwave results for scMix data: 
NUM_COMPONENTS = 10
file_name = '/home/delaram/sciRED//review_analysis/benchmark_methods/zinbwave_scores_scMix_numcomp_'+str(NUM_COMPONENTS)+'.csv'
zinbwave_scores = pd.read_csv(file_name, index_col=0)
print(zinbwave_scores.shape)

zinbwave_scores.index = data.obs.index.values
zinbwave_scores.head()
### check if cell_line column in zinbwave_scores and metadata are the same
print(np.all(data.obs['cell_line'] == zinbwave_scores['cell_line']))

y_protocol_zinbwave = zinbwave_scores['sample']
y_cell_line_zinbwave = zinbwave_scores['cell_line']

### extract columns with names as W* from zinbwave_scores
pattern_cols = [col for col in zinbwave_scores.columns if 'W' in col]
zinbwave_scores_factor = zinbwave_scores[pattern_cols]
## convert to numpy.ndarray
zinbwave_scores_factor = zinbwave_scores_factor.to_numpy()

### run FCAT on zinbwave results
fcat_protocol = efca.FCAT(y_protocol_zinbwave, zinbwave_scores_factor, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line_zinbwave, zinbwave_scores_factor, scale='standard', mean='arithmatic')
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)

vis.plot_FCAT(fcat, title='', color='coolwarm',
                x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
                x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

title = 'zinbwave on scMix data'
### make a dictionary of colors for each sample in y_sample
vis.plot_pca(zinbwave_scores_factor, NUM_COMP_TO_VIS, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title=title,
               plt_legend_list=plt_legend_cell_line)

vis.plot_pca(zinbwave_scores_factor, NUM_COMP_TO_VIS,
                cell_color_vec= colors_dict_scMix['protocol'], 
                legend_handles=True,
                title=title,
                plt_legend_list=plt_legend_protocol)