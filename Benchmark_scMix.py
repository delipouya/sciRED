#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from exutils import permutation as pmut
import numpy as np
import pandas as pd

from sciRED import rotations as rot
from sciRED.utils import preprocess as proc
from sciRED import glm
from exutils import ex_preprocess as exproc
from exutils import ex_visualize as exvis
import time

np.random.seed(10)
NUM_COMPONENTS = 30
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5

data_file_path = '/home/delaram/sciFA/Data/scMix_3cl_merged.h5ad'
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
#### design matrix - library size only
x = proc.get_library_design_mat(data, lib_size='nCount_originalexp')
glm_fit_dict = glm.poissonGLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
resid_dict = {'pearson': resid_pearson}

residual_type = 'pearson'
### make a for loop to calculate the importance scores for each residual type
fcat_dict = {}
time_dict_a_level_dict = {}


for residual_type in resid_dict.keys():
    print('--'*20)
    print('residual type: ', residual_type)
    resid = resid_dict[residual_type]
    y = resid.T 
    print('y shape: ', y.shape) # (num_cells, num_genes)

    pipeline = Pipeline([('scaling', StandardScaler()), 
                         ('pca', PCA(n_components=NUM_COMPONENTS))])
    pca_scores = pipeline.fit_transform(y)
    pca = pipeline.named_steps['pca']
    pca_loading = pca.components_

    ####### Applying varimax rotation to the factor scores
    rotation_results_varimax = rot.varimax(pca_loading.T)
    varimax_loading = rotation_results_varimax['rotloading']
    pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

    factor_scores = pca_scores_varimax
    scores_included = 'baseline'
    #n = 1000
    n = 500

    ####################################
    #### fcat score calculation and run time as baseline ######
    fcat_dict_protocol, time_dict_a_level_dict_protocol = pmut.get_FCAT_dict(y_protocol, factor_scores, time_eff=False) 
    fcat_dict_cell_line, time_dict_a_level_dict_cell_line = pmut.get_FCAT_dict(y_cell_line, factor_scores, time_eff=False) 
    covariate_list = ['protocol']*3 + ['cell_line']*3


    ##########################################################
    ########### Comparing model run times
    time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
    ########## Comparing time differences between models
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', 
                                    columns=list(time_df_dict.values())[0].keys())
    #pmut.plot_runtime_barplot(time_df)

    fcat_dict = {**fcat_dict_protocol, **fcat_dict_cell_line}
    fcat_m = pmut.get_melted_fcat(fcat_dict)
    fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]

    ### save baseline (unshuffled) fcat_m to csv
    fcat_m.to_csv('/home/delaram/sciFA/Results/benchmark/'+residual_type+'/base/'+
                 'importance_df_melted_scMixology_'+residual_type+'_'+'baseline.csv')

    t_start_total = time.time()
    #### shuffle the covariate vectors n times in a loop
    for i in range(n):
        print('i: ', i)

        y_protocol_shuffled = pmut.shuffle_covariate(y_protocol)
        y_cell_line_shuffled = pmut.shuffle_covariate(y_cell_line)

        ####################################
        #### Importance calculation and run time for model comparison  ######
        ####################################
        fca_dict_protocol, time_dict_a_level_dict_protocol =  pmut.get_FCAT_dict(y_protocol_shuffled, factor_scores, time_eff=False) 
        fca_dict_cell_line, time_dict_a_level_dict_cell_line =  pmut.get_FCAT_dict(y_cell_line_shuffled, factor_scores, time_eff=False) 
        
        ####################################
        ########### Comparing model run times
        ####################################
        time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
        ########## Comparing time differences between models
        time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
        #plot_runtime_barplot(time_df)
        time_df.to_csv('/home/delaram/sciFA/Results/benchmark/'+residual_type+
                       '/time/' + 'time_df_scMixology_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')
        
        ############################################################
        ########## Comparing factor scores between models
        ############################################################
        #### merge two fca_dict_protocol and fca_dict_cell_line dicts
        fca_dict = {**fca_dict_protocol, **fca_dict_cell_line}
        fca_m = pmut.get_melted_fcat(fca_dict)
        fca_m['residual_type'] = [residual_type]*fca_m.shape[0]

        ### save pmututated fca_m to csv
        fca_m.to_csv('/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/imp/'+
                               'importance_df_melted_scMixology_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')

    t_end_total = time.time()
    print('Total time: ', t_end_total - t_start_total)


