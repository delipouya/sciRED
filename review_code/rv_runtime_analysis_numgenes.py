
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
import time


### define a function to subset the data to include num_cells
def subset_data_cells(data, num_cells):
    metadata = data.obs
    idx = random.sample(range(0, data.shape[0]), num_cells)
    
    data = data[idx, :]
    ### subset metadata based on the idx
    metadata = metadata.iloc[idx]

    return data, metadata


def get_nonzero_data(data) -> 'AnnData':    
    ''' subset the data matrix to the top num_genes genes
    y: numpy array of the gene expression matrix (n_cells, n_genes)
    random: whether to randomly select the genes or select the genes with highest variance
    num_genes: number of genes to select
    '''
    data_numpy = data.X.toarray()
    cell_sums = np.sum(data_numpy,axis=1) # row sums - library size
    gene_sums = np.sum(data_numpy,axis=0) # col sums - sum reads in a gene
    data = data[cell_sums!=0,gene_sums != 0] ## cells, genes
    ### subset the data matrix to the top num_genes genes
    return data


def subset_data_genes(data, num_genes):
    ''' 
    subset the data matrix to the top num_genes genes
    y: numpy array of the gene expression matrix (n_cells, n_genes)
    random: whether to randomly select the genes or select the genes with highest variance
    num_genes: number of genes to select
    '''
    data_numpy = data.X.toarray()
    ### calculate the variance for each gene
    gene_vars = np.var(data_numpy, axis=0)
    ### select the top num_genes genes with the highest variance
    gene_idx = np.argsort(gene_vars)[-num_genes:]

    #### select num_genes genes based on variance
    ## sort the gene_idx in ascending order
    gene_idx = np.sort(gene_idx)
    data = data[:,gene_idx]

    return data


random.seed(0)
NUM_COMPONENTS = 30
NUM_COMP_TO_VIS = 5
NUM_CELLS = 40000

data_file_path = '/home/delaram/sciRED/review_analysis/Ischemia_Reperfusion_Responses_Human_Lung_Transplants.h5ad'
data = exproc.import_AnnData(data_file_path)

data = get_nonzero_data(data) # subset the data to num_genes HVGs
data, metadata = subset_data_cells(data, num_cells=NUM_CELLS)
data_orig = data.copy()

y_lib_size = metadata[['nCount_RNA']].squeeze()
y_timepoint = metadata[['timepoint']].squeeze()
y_phase = metadata[['Phase']].squeeze()

y_case = metadata[['ltx_case']].squeeze()
y_sample = metadata[['sample_name']].squeeze()
y_donor_id = metadata[['donor_id']].squeeze()
y_recipient_origin = metadata[['recipient_origin']].squeeze()
y_cell_type = metadata[['cell_type']].squeeze()
y_tissue = metadata[['tissue']].squeeze()
y_sex = metadata[['sex']].squeeze()

#### design matrix - library size and sample
column_levels = y_case.unique() 
dict_covariate = {}
for column_level in column_levels:
    dict_covariate[column_level] = proc.get_binary_covariate(y_case.squeeze(), column_level)

#### stack colummns of dict_covariate 
x_sample = np.column_stack(([dict_covariate[column] for column in column_levels]))

x = np.column_stack((y_lib_size, x_sample)) 
x = sm.add_constant(x) ## adding the intercept



data_orig = data.copy()
# make a dictionary to save the run time for each number of cells
run_time_dict = {}

## loop to calculate the time needed to include 10K, 5K, 2K, 500 HVGs
for NUM_genes in [10000, 5000, 2000, 500]:
    data = subset_data_genes(data_orig, num_genes=NUM_genes)
    y, genes, num_cells, num_genes = proc.get_data_array(data)
    ## print shape of the data
    print(f'Number of cells: {num_cells}, Number of genes: {num_genes}')

    ####################################
    #### fit GLM to each gene ######
    ####################################

    ### start the timer
    start_time = time.time()

    ### fit GLM to each gene
    glm_fit_dict = glm.poissonGLM(y, x)
    resid_pearson = glm_fit_dict['resid_pearson'] 
    y = resid_pearson.T # (num_cells, num_genes)

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

    ####################################
    #### Matching between factors and covariates ######
    ####################################

    ######## Applying varimax rotation to the factor scores
    rotation_results_varimax = rot.varimax(pca_loading.T)
    varimax_loading = rotation_results_varimax['rotloading']
    pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])
    factor_scores = pca_scores_varimax

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

    ### end the timer
    end_time = time.time()
    ### record the run time
    print(f'Run time for {NUM_genes} genes: {end_time-start_time} seconds')
    run_time_dict[NUM_genes] = end_time-start_time
    


### save run_time_dict to a csv file
run_time_df = pd.DataFrame(run_time_dict.items(), columns=['num_genes', 'run_time'])
run_time_df.to_csv('/home/delaram/sciRED/review_analysis/run_time_dict_lung_genenum.csv')

### plot the run time
plt.plot(run_time_dict.keys(), run_time_dict.values())
plt.xlabel('Number of genes')
plt.ylabel('Run time (seconds)')
plt.title('Run time for different number of genes')
## save the plot
plt.savefig('/home/delaram/sciRED/review_analysis/run_time_lung_genenum.png')
plt.show()


### read the csv and re-create the plot with dots as well as lines
run_time_df = pd.read_csv('/home/delaram/sciRED/review_analysis/run_time_dict_lung_genenum.csv')
plt.plot(run_time_df['num_genes'], run_time_df['run_time'], 'o-')
plt.xlabel('Number of genes', fontsize=15)
plt.ylabel('Run time (seconds)', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Run time (seconds)')
plt.title('Run time for different number of genes', fontsize=15)
plt.show()
