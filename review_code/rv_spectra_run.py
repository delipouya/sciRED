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


import Spectra
from Spectra import K_est as kst
import cytopus as cp
import time


np.random.seed(10)
#NUM_COMPONENTS = 30
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5


data_file_path = '/home/delaram/sciFA/Data/PBMC_Lupus_Kang8vs8_data.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)
y_sample, y_stim, y_cell_type, y_cluster  = exproc.get_metadata_humanPBMC(data)


G = cp.KnowledgeBase()
G.celltypes #list of all cell types in KnowledgeBase
currect_labels = y_cell_type.unique()
available_labels = G.celltypes

# Matching dictionary
matching_dict = {
    'CD4 T cells': 'CD4-T',
    'CD14+ Monocytes': 'mono',
    'Dendritic cells': 'cDC',
    'NK cells': 'NK',
    'CD8 T cells': 'CD8-T',
    'B cells': 'B',
    'Megakaryocytes': 'all-cells',
    'FCGR3A+ Monocytes': 'mono',
    'NA': 'all-cells'
}

replaced_labels = [matching_dict.get(label, label) for label in y_cell_type]
data.obs['replaced_labels'] = replaced_labels

### use values of matching_dict as celltype_of_interest
celltype_of_interest = list(matching_dict.values())
G.get_celltype_processes(celltype_of_interest,
                            global_celltypes = ['all-cells'],
                         get_children=True,
                         get_parents =False)

annotations = G.celltype_process_dict
cell_type_keys = 'replaced_labels'
#L = kst.estimate_L(data, attribute = cell_type_keys, highly_variable = False)


# Number of factors per cell type: {'CD4-T': 1, 'mono': 1, 'cDC': 1, 'NK': 1, 'CD8-T': 1, 'B': 1, 'all-cells': 1}
num_per_cell_factors = 5
L = {cell_type: num_per_cell_factors for cell_type in matching_dict.values()}
L['global'] = num_per_cell_factors
## sum values of L
L_sum = sum(L.values())
NUM_COMPONENTS = L_sum

data.shape #(29065, 2000) - an hour
#   6%|▋         | 646/10000 [04:09<1:02:38,  2.49it/s]

### time the model fitting
start = time.time()
model = Spectra.est_spectra(
    adata=data, 
    gene_set_dictionary=annotations, 
    use_highly_variable=False,
    cell_type_key=cell_type_keys, 
    use_weights=True,
    lam=0.1, #varies depending on data and gene sets, try between 0.5 and 0.001
    delta=0.001, 
    kappa=None,
    rho=0.001, 
    L = L,
    use_cell_types=True, #False
    n_top_vals=50,
    label_factors=False, #True, 
    overlap_threshold=0.2,
    clean_gs = True, 
    min_gs_num = 3,
    num_epochs=10000
)

end = time.time()
print('Time to fit the model: ', end-start)
print('Time to fit the model in minutes: ', (end-start)/60)


## time to fit the model for 128 components
# Time to fit the model:  1672.7507145404816
# Time to fit the model in minutes:  27.879178575674693

factors = data.uns['SPECTRA_factors'] # factors x genes matrix that tells you how important each gene is to the resulting factors
markers = data.uns['SPECTRA_markers'] # factors x n_top_vals list of n_top_vals top markers per factor
cell_scores = data.obsm['SPECTRA_cell_scores'] # cells x factors matrix of cell scores
#vocab = data.var['spectra_vocab'] # boolean matrix of size # of genes that indicates the set of genes used to fit spectra 

### save the factors, markers, cell_scores, and vocab to a csv file
factors_df = pd.DataFrame(factors.T)
factors_df.columns = ['F'+str(i) for i in range(1, factors_df.shape[1]+1)]
factors_df.index = genes

markers_df = pd.DataFrame(markers.T)
markers_df.columns = ['F'+str(i) for i in range(1, markers_df.shape[1]+1)]

cell_scores_df = pd.DataFrame(cell_scores)
cell_scores_df.columns = ['F'+str(i) for i in range(1, cell_scores_df.shape[1]+1)]
cell_scores_df.index = data.obs_names

cell_scores_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/spectra_cell_scores_pbmc_numcomp_'+ str(NUM_COMPONENTS)+ '.csv')
factors_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/spectra_factors_pbmc_numcomp_'+ str(NUM_COMPONENTS)+ '.csv')
markers_df.to_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/spectra_markers_pbmc_numcomp_'+ str(NUM_COMPONENTS)+ '.csv')



