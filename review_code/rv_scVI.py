import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.pipeline import Pipeline

from sciRED import ensembleFCA as efca
from sciRED import rotations as rot
from sciRED import metrics as met

from sciRED.utils import preprocess as proc
from sciRED.utils import visualize as vis
from sciRED.utils import corr
from sciRED.examples import ex_preprocess as exproc
from sciRED.examples import ex_visualize as exvis

import scvi
import scanpy as sc
import time

np.random.seed(10)
NUM_COMPONENTS = 10
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5

###############################################
############### scMix ############################
################################################

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


start = time.time()
# Prepare the AnnData object for scVI
scvi.model.SCVI.setup_anndata(data)

# Define and train the scVI model
model = scvi.model.SCVI(data, n_latent=NUM_COMPONENTS)
model.train()
latent_embeddings = model.get_latent_representation()
end = time.time()

print('Time to fit scVI - numcomp '+ str(NUM_COMPONENTS)+ ' : ', end-start)
print('Time to fit scVI (min) - numcomp '+ str(NUM_COMPONENTS)+ ' : ', (end-start)/60)


data.obsm['X_scVI'] = latent_embeddings  # Store in AnnData object
print(latent_embeddings.shape)

### save the scvi_scores_df a csv file
metadata = data.obs
scvi_scores_df = pd.DataFrame(latent_embeddings)
scvi_scores_df.columns = ['F'+str(i) for i in range(1, scvi_scores_df.shape[1]+1)]
if scvi_scores_df.shape[0] == metadata.shape[0]:
    scvi_scores_df_merged = pd.concat([metadata.reset_index(drop=True), 
                                      scvi_scores_df.reset_index(drop=True)], axis=1)
else:
    raise ValueError("Number of rows in 'metadata' does not match 'scvi_scores_df'.")
scvi_scores_df_merged.head()

scvi_scores_df_merged.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/scvi_scores_scMix_numcomp_'+
                             str(NUM_COMPONENTS)+'.csv')

factor_scores = latent_embeddings
### FCAT needs to be calculated for each covariate separately
fcat_protocol = efca.FCAT(y_protocol, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line, factor_scores, scale='standard', mean='arithmatic')
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

title = 'scVI on count data'
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






###############################################
######################## Human PBMC ############################
################################################

data_file_path = '/home/delaram/sciFA/Data/PBMC_Lupus_Kang8vs8_data.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)
y_sample, y_stim, y_cell_type, y_cluster  = exproc.get_metadata_humanPBMC(data)

colors_dict_humanPBMC = exvis.get_colors_dict_humanPBMC(y_sample, y_stim, y_cell_type)
plt_legend_sample = exvis.get_legend_patch(y_sample, colors_dict_humanPBMC['sample'] )
plt_legend_stim = exvis.get_legend_patch(y_stim, colors_dict_humanPBMC['stim'] )
plt_legend_cell_type = exvis.get_legend_patch(y_cell_type, colors_dict_humanPBMC['cell_type'] )
####################################

data = data.copy()
###############################################


start = time.time()
# Prepare the AnnData object for scVI
scvi.model.SCVI.setup_anndata(data)

# Define and train the scVI model
model = scvi.model.SCVI(data, n_latent=NUM_COMPONENTS)
model.train()
latent_embeddings = model.get_latent_representation()
end = time.time()

print('Time to fit scVI - numcomp '+ str(NUM_COMPONENTS)+ ' : ', end-start)
print('Time to fit scVI (min) - numcomp '+ str(NUM_COMPONENTS)+ ' : ', (end-start)/60)

data.obsm['X_scVI'] = latent_embeddings  # Store in AnnData object
print(latent_embeddings.shape)

### save the scvi_scores_df a csv file
metadata = data.obs
scvi_scores_df = pd.DataFrame(latent_embeddings)
scvi_scores_df.columns = ['F'+str(i) for i in range(1, scvi_scores_df.shape[1]+1)]

if scvi_scores_df.shape[0] == metadata.shape[0]:
    scvi_scores_df_merged = pd.concat([metadata.reset_index(drop=True), 
                                      scvi_scores_df.reset_index(drop=True)], axis=1)
else:
    raise ValueError("Number of rows in 'metadata' does not match 'scvi_scores_df'.")
scvi_scores_df_merged.head()

scvi_scores_df_merged.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/scvi_scores_pbmc_numcomp_'+
                             str(NUM_COMPONENTS)+'.csv')

factor_scores = latent_embeddings

### FCAT needs to be calculated for each covariate separately
fcat_sample = efca.FCAT(y_sample, factor_scores, scale='standard', mean='arithmatic')
fcat_stim = efca.FCAT(y_stim, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_type = efca.FCAT(y_cell_type, factor_scores, scale='standard', mean='arithmatic')


### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_sample, fcat_stim, fcat_cell_type], axis=0)

#fcat = pd.concat([fcat_stim, fcat_cell_type], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table

vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)



title = 'scVI on count data'
NUM_COMP_TO_VIS = 3