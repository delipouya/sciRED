'''
This script is used to benchmark the performance of sciRED on the Human Liver Atlas dataset.
we generate the baseline and shuffle importance scores for the residual types: pearson, response, and deviance.
The importance scores are calculated using the FCAT method.
The benchmarking is done by comparing the run time of the models and the importance scores.
the mean importance scores for three scaling and two mean types are calculated for each residual type.
The importance scores and the mean importance scores are saved in csv files.
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


data_file_path = '/home/delaram/sciFA//Data/HumanLiverAtlas.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)
y_sample, y_cell_type = exproc.get_metadata_humanLiver(data)

num_levels_sample = len(y_sample.unique())
num_levels_cell_type = len(y_cell_type.unique())

x = proc.get_library_design_mat(data, lib_size='total_counts')
glm_fit_dict = glm.poissonGLM(y, x)

#### extracting the pearson residuals, response residuals and deviance residuals
resid_pearson = glm_fit_dict['resid_pearson'] 
resid_response = glm_fit_dict['resid_response']
resid_deviance = glm_fit_dict['resid_deviance']
resid_dict = {'pearson': resid_pearson, 'response': resid_response, 'deviance': resid_deviance}


### make a for loop to calculate the importance scores for each residual type
fcat_dict = {}
time_dict_a_level_dict = {}


for residual_type in resid_dict.keys():
    print('--'*20)
    print('residual type: ', residual_type)
    resid = resid_dict[residual_type]
    y = resid.T # (num_cells, num_genes)

    ### using pipeline to scale the gene expression data first
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
    n = 500

    ####################################
    #### fcat score calculation and run time as baseline ######
    fcat_dict_sample, time_dict_a_level_dict_sample = pmut.get_FCAT_dict(y_sample, factor_scores, time_eff=True) 
    fcat_dict_cell_type, time_dict_a_level_dict_cell_type = pmut.get_FCAT_dict(y_cell_type, factor_scores, time_eff=True) 

    meanimp_standard_arith_sample, meanimp_standard_geom_sample, meanimp_minmax_arith_sample, meanimp_minmax_geom_sample, meanimp_rank_arith_sample, meanimp_rank_geom_sample = pmut.get_mean_fcat_list(fcat_dict_sample)
    meanimp_standard_arith_cell_type, meanimp_standard_geom_cell_type, meanimp_minmax_arith_cell_type, meanimp_minmax_geom_cell_type, meanimp_rank_arith_cell_type, meanimp_rank_geom_cell_type = pmut.get_mean_fcat_list(fcat_dict_cell_type)

    meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_sample, meanimp_standard_arith_cell_type], axis=0)
    meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_sample, meanimp_standard_geom_cell_type], axis=0)
    meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_sample, meanimp_minmax_arith_cell_type], axis=0)
    meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_sample, meanimp_minmax_geom_cell_type], axis=0)
    meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_sample, meanimp_rank_arith_cell_type], axis=0)
    meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_sample, meanimp_rank_geom_cell_type], axis=0)


    meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                    meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                    meanimp_rank_arith_df, meanimp_rank_geom_df]

    mean_type_list = ['arithmatic', 'geometric', 
                    'arithmatic', 'geometric', 
                    'arithmatic', 'geometric']

    scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']

    scores_included_list = [scores_included]*len(meanimp_df_list)
    covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type

    meanimp_df = pmut.concatMeanFCAT(meanimp_df_list, mean_type_list, 
                                        scale_type_list, scores_included_list, 
                                        residual_type=residual_type, covariate_list=covariate_list)


    ############################################################
    ########### Comparing model run times
    time_df_dict = {**time_dict_a_level_dict_sample, **time_dict_a_level_dict_cell_type}
    ########## Comparing time differences between models
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', 
                                    columns=list(time_df_dict.values())[0].keys())
    #flabel.plot_runtime_barplot(time_df)

    ########## Comparing factor scores between models
    fcat_dict = {**fcat_dict_sample, **fcat_dict_cell_type}
    fcat_m = pmut.get_melted_fcat(fcat_dict)
    fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]


    ### save importance_df_m and meanimp_df to csv
    fcat_m.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+
                  '/base/'+'importance_df_melted_human_liver_'+residual_type+'_'+'baseline.csv')
    meanimp_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+
                      '/base/'+'meanimp_df_'+'human_liver_'+residual_type+'_'+'baseline.csv')

    t_start_total = time.time()
    #### shuffle the covariate vectors n times in a loop
    for i in range(n):
        print('i: ', i)

        y_sample_shuffled = pmut.shuffle_covariate(y_sample)
        y_cell_type_shuffled = pmut.shuffle_covariate(y_cell_type)

        ####################################
        #### Importance calculation and run time for model comparison  ######
        ####################################    
        fcat_dict_sample, time_dict_a_level_dict_sample = pmut.get_FCAT_dict(y_sample, factor_scores, time_eff=True) 
        fcat_dict_cell_type, time_dict_a_level_dict_cell_type = pmut.get_FCAT_dict(y_cell_type, factor_scores, time_eff=True) 

        ####################################
        ########### Comparing model run times
        ####################################
        time_df_dict = {**time_dict_a_level_dict_sample, **time_dict_a_level_dict_cell_type}
        ########## Comparing time differences between models
        time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
        #plot_runtime_barplot(time_df)
        ### save time_df to csv
        time_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+
                       '/time/' + 'time_df_human_liver_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')
        
        ####################################
        ##### Mean importance calculation ########
        meanimp_standard_arith_sample, meanimp_standard_geom_sample, meanimp_minmax_arith_sample, meanimp_minmax_geom_sample, meanimp_rank_arith_sample, meanimp_rank_geom_sample = pmut.get_mean_fcat_list(fcat_dict_sample)
        meanimp_standard_arith_cell_type, meanimp_standard_geom_cell_type, meanimp_minmax_arith_cell_type, meanimp_minmax_geom_cell_type, meanimp_rank_arith_cell_type, meanimp_rank_geom_cell_type = pmut.get_mean_fcat_list(fcat_dict_cell_type)


        meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_sample, meanimp_standard_arith_cell_type], axis=0)
        meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_sample, meanimp_standard_geom_cell_type], axis=0)
        meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_sample, meanimp_minmax_arith_cell_type], axis=0)
        meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_sample, meanimp_minmax_geom_cell_type], axis=0)
        meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_sample, meanimp_rank_arith_cell_type], axis=0)
        meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_sample, meanimp_rank_geom_cell_type], axis=0)


        meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                        meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                        meanimp_rank_arith_df, meanimp_rank_geom_df]

        mean_type_list = ['arithmatic', 'geometric', 
                        'arithmatic', 'geometric', 
                        'arithmatic', 'geometric']

        scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']

        scores_included = 'shuffle'#'baseline'#'top_cov' 'top_FA' 
        scores_included_list = [scores_included]*len(meanimp_df_list)
        covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type

        meanimp_df = pmut.concatMeanFCAT(meanimp_df_list, mean_type_list, 
                                            scale_type_list, scores_included_list, 
                                            residual_type=residual_type, covariate_list=covariate_list)

        ############################################################
        ########## Comparing factor scores between models
        ############################################################
        #### merge two importance_df_dict_sample and importance_df_dict_cell_type dicts
        fcat_dict = {**fcat_dict_sample, **fcat_dict_cell_type}
        fcat_m = pmut.get_melted_fcat(fcat_dict)
        ### add a column for residual type name to importance_df_m
        fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]


        ### save importance_df_m and meanimp_df to csv
        fcat_m.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/shuffle/imp/'+
                               'importance_df_melted_human_liver_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')
        meanimp_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/shuffle/meanimp/'+
                          'meanimp_df_'+'human_liver_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')


    t_end_total = time.time()
    print('Total time: ', t_end_total - t_start_total)


