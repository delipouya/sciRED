## This script is to analyze the run time of sciRED for different number of cells

# The script should output the following:
# - a plot showing the run time for different number of cells
# - a csv file containing the run time for different number of cells

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
def subset_data(data, num_cells):
    data = data_orig.copy()
    metadata = data.obs
    idx = random.sample(range(0, data.shape[0]), num_cells)
    
    data = data[idx, :]
    ### subset metadata based on the idx
    metadata = metadata.iloc[idx]

    return data, metadata

random.seed(0)
NUM_GENES = 2000
NUM_COMPONENTS = 30
NUM_COMP_TO_VIS = 5

data_file_path = '/home/delaram/sciRED/review_analysis/Ischemia_Reperfusion_Responses_Human_Lung_Transplants.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
data_orig = data.copy()

# make a dictionary to save the run time for each number of cells
run_time_dict = {}


### write a loop to run the following code for 100K, 80K, 60K, 40K, 20K cells and record the run time
for num_cells in [100000, 80000, 60000, 40000, 20000]:

    data, metadata = subset_data(data_orig, num_cells=num_cells)
    
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
        
    y, genes, num_cells, num_genes = proc.get_data_array(data)

    ####################################
    #### fit GLM to each gene ######
    ####################################

    #### design matrix - library size and sample
    column_levels = y_case.unique() 
    dict_covariate = {}
    for column_level in column_levels:
        dict_covariate[column_level] = proc.get_binary_covariate(y_case.squeeze(), column_level)
    #### stack colummns of dict_covariate 
    x_sample = np.column_stack(([dict_covariate[column] for column in column_levels]))

    x = np.column_stack((y_lib_size, x_sample)) 
    x = sm.add_constant(x) ## adding the intercept

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
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=NUM_COMPONENTS))])
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
    print(f'Run time for {num_cells} cells: {end_time-start_time} seconds')
    run_time_dict[num_cells] = end_time-start_time


print(run_time_dict)
### plot the run time
plt.plot(run_time_dict.keys(), run_time_dict.values())
plt.xlabel('Number of cells')
plt.ylabel('Run time (seconds)')
plt.title('Run time for different number of cells')
## save the plot
plt.savefig('/home/delaram/sciRED/review_analysis/run_time_lung_cellnum.png')
plt.show()
## save the run time dictionary as a csv file
df = pd.DataFrame(run_time_dict.items(), columns=['num_cells', 'run_time'])
df.to_csv('/home/delaram/sciRED/review_analysis/run_time_dict_lung_cellnum.csv')


### read the csv file
df = pd.read_csv('/home/delaram/sciRED/review_analysis/run_time_dict_lung_cellnum.csv')
### re-create the plot with dots as well as lines
plt.plot(df['num_cells'], df['run_time'], 'o-')
plt.xlabel('Number of cells', fontsize=14)
plt.ylabel('Run time (seconds)', fontsize=14)
plt.title('Run time for different number of cells', fontsize=16)
## make the x and y axis labels font size larger
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
plt.show()



