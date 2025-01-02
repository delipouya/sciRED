
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sciRED import ensembleFCA as efca
from sciRED import glm
from sciRED import rotations as rot
from sciRED import metrics as met

from sciRED.utils import preprocess as proc
from sciRED.utils import visualize as vis
from sciRED.utils import corr
from sciRED.examples import ex_preprocess as exproc
from sciRED.examples import ex_visualize as exvis

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


scmix_usage = '/home/delaram/sciRED/review_analysis/benchmark_methods/cNMF/scMix_cNMF_numcomp_10_usage_1030.csv'
scmix_usage = '/home/delaram/sciRED/review_analysis/benchmark_methods/cNMF/scMix_cNMF_v2_numcomp_30_usage.csv'

usage = pd.read_csv(scmix_usage, index_col=0)
factor_scores = usage.to_numpy()

### FCAT needs to be calculated for each covariate separately
fcat_protocol = efca.FCAT(y_protocol, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_line = efca.FCAT(y_cell_line, factor_scores, scale='standard', mean='arithmatic')

### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_protocol, fcat_cell_line], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table

vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

