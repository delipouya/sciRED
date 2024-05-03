#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate a factor with multiple covariates and calculate the overlap between the covariates
and the matching scores between the covariates and the factor
calculate the correlation between the overlap and the matching scores
repeat the simulation nn times and calculate the average correlation between the overlap and all of the scores
save the results in a csv file
"""
import numpy as np
import pandas as pd

from sciRED.utils import preprocess as proc
from exutils import ex_preprocess as exproc
import time
import numpy as np
import pandas as pd

from sciRED import ensembleFCA as efca
from sciRED import glm
from sciRED import rotations as rot
from sciRED import metrics as met

from sciRED.utils import preprocess as proc
from sciRED.utils import visualize as vis
from sciRED.utils import corr
from exutils import ex_preprocess as exproc
from exutils import ex_visualize as exvis
from exutils import simulation as sim

np.random.seed(0)

num_sim_rounds = 100
#num_sim_rounds = 2
num_factors = 20
num_mixtures = 2 ## each gaussian represents a covariate level 
num_samples = 10000
#### perform the simulation n times and calculate the average correlation between the overlap and all of the scores
corr_df = pd.DataFrame()
corr_df_list = []

### calculate the time of the simulation
start_time = time.time()


for i in range(num_sim_rounds):
    
    start_time_in = time.time()
    print('simulation: ', i)
    sim_factors_list = []
    overlap_mat_list = []
    covariate_list = []

    for i in range(num_factors):
        a_random_factor, overlap_matrix, mu_list, sigma_list, p_list = sim.get_simulated_factor_object(n=num_samples, num_mixtures=num_mixtures, 
                                                                                                mu_min=0, mu_max=None,
                                                                                                sigma_min=0.5, sigma_max=1, p_equals=True)  
        sim_factors_list.append(sim.unlist(a_random_factor))
        overlap_mat_list.append(overlap_matrix)
        covariate_list.append(sim.get_sim_factor_covariates(a_random_factor))

    overlap_mat_flat = sim.convert_matrix_list_to_vector(overlap_mat_list) ### flatten the overlap matrix list

    ### convert sim_factors_list to a numpy nd array with shape (num_samples, num_factors)
    sim_factors_array = np.asarray(sim_factors_list).T
    sim_factors_df = pd.DataFrame(sim_factors_array, columns=['factor'+str(i+1) for i in range(num_factors)])
    factor_scores = sim_factors_array
    covariate_vector = pd.Series(covariate_list[0])

    ### calculate the mean importance of each covariate level
    mean_importance_df = efca.get_mean_importance_all_levels(covariate_vector, factor_scores)
    all_covariate_levels = mean_importance_df.index.values

    #################################### 
    #### calculate the overlap and matching scores for all the factors
    #################################### 
    match_score_mat_meanImp_list = []

    for i in range(num_factors): ## i is the factor index
        match_score_mat_meanImp_list.append(sim.get_pairwise_match_score_matrix(mean_importance_df,i))
    match_score_mat_meanImp_flat = sim.convert_matrix_list_to_vector(match_score_mat_meanImp_list)

    silhouette_scores, calinski_harabasz_scores, wvrs,\
        davies_bouldin_scores = met.get_kmeans_scores(factor_scores, time_eff=False)
    

    #silhouette_scores_gmm, vrs_gmm, wvrs_gmm = met.get_gmm_scores(factor_scores, time_eff=False)
    #likelihood_ratio_scores = met.get_likelihood_ratio_test_all(factor_scores)
    bimodality_index_scores = met.get_bimodality_index_all(factor_scores)
    dip_scores, pval_scores = met.get_dip_test_all(factor_scores)
    #kurtosis_scores = met.get_factor_kurtosis_all(factor_scores)
    #outlier_sum_scores = met.get_outlier_sum_statistic_all(factor_scores)

    bimodality_metrics = ['calinski_harabasz', 'davies_bouldin',
                           'silhouette', 'wvrs', 'bimodality_index', 'dip_score']
    bimodality_scores = [ calinski_harabasz_scores, davies_bouldin_scores,
                                        silhouette_scores, wvrs,
                                        bimodality_index_scores,
                                        dip_scores]

    #### Scaled variance
    SV_all_factors = met.get_factors_SV_all_levels(factor_scores, covariate_vector)

    ### label dependent factor metrics - Homogeneity
    ASV_all_arith = met.get_ASV_all(factor_scores, covariate_vector, mean_type='arithmetic')
    ASV_all_geo = met.get_ASV_all(factor_scores, covariate_vector, mean_type='geometric')
    ASV_simpson = met.get_all_factors_simpson(pd.DataFrame(met.get_factors_SV_all_levels(factor_scores, covariate_vector)))
    ### calculate ASV based on entropy on the scaled variance per covariate for each factor
    ASV_entropy = met.get_factor_entropy_all(pd.DataFrame(met.get_factors_SV_all_levels(factor_scores, covariate_vector)))

    ### calculate diversity metrics (specificity)
    factor_gini_meanimp = met.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
    factor_simpson_meanimp = met.get_all_factors_simpson_D_index(mean_importance_df) ## calculated for each factor in the importance matrix
    factor_entropy_meanimp = met.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix

    #### label free factor metrics
    factor_variance_all = met.get_factor_variance_all(factor_scores)

    ### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
    AUC_all_factors_df = met.get_AUC_all_factors_df(factor_scores, covariate_vector)
    AUC_all_factors_df_1 = met.get_reversed_AUC_df(AUC_all_factors_df)
    ### calculate arithmatic and geometric mean of each factor using AUC_all_factors_df_1
    AUC_1_arith_mean = sim.get_arithmatic_mean_df(AUC_all_factors_df_1)
    AUC_1_geo_mean = sim.get_geometric_mean_df(AUC_all_factors_df_1)

    #### calculate the correlation between the overlap and all of the scores and save in a dataframe
    corr_df_temp = pd.DataFrame()
    for i in range(len(bimodality_metrics)):
        corr_df_temp[bimodality_metrics[i]] = [np.corrcoef(overlap_mat_flat, bimodality_scores[i])[0,1]]

    
    corr_df_temp['ASV_arith'] = [np.corrcoef(overlap_mat_flat, ASV_all_arith)[0,1]]
    corr_df_temp['ASV_geo'] = [np.corrcoef(overlap_mat_flat, ASV_all_geo)[0,1]]

    corr_df_temp['ASV_simpson'] = [np.corrcoef(overlap_mat_flat, ASV_simpson)[0,1]]
    corr_df_temp['ASV_entropy'] = [np.corrcoef(overlap_mat_flat, ASV_entropy)[0,1]]
    
    corr_df_temp['1-AUC_arith'] = [np.corrcoef(overlap_mat_flat, AUC_1_arith_mean)[0,1]]
    corr_df_temp['1-AUC_geo'] = [np.corrcoef(overlap_mat_flat, AUC_1_geo_mean)[0,1]]

    corr_df_temp['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance_all)[0,1]]

    ## only include in case #covariate levels > 3 - gini is a single value cant be saved in corr_df_temp
    corr_df_temp['factor_entropy_meanImp'] = [np.corrcoef(overlap_mat_flat, factor_entropy_meanimp)[0,1]]
    corr_df_temp['factor_simpon_meanImp'] = [np.corrcoef(overlap_mat_flat, factor_simpson_meanimp)[0,1]]

    corr_df_temp['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance_all)[0,1]]
        
    corr_df_temp = corr_df_temp.T
    corr_df_temp.columns = ['overlap']
    #corr_df_temp = corr_df_temp.sort_values(by='overlap', ascending=False)
    corr_df_list.append(corr_df_temp)

    end_time_in = time.time()
    print('simulation: ', str(i), ' time: ', end_time_in - start_time_in)


end_time = time.time()
print('time: ', end_time - start_time)
### convert time to minutes
time_minutes = round((end_time - start_time)/60, 2)
print('time in minutes: ', time_minutes)

### calculate the average correlation between the overlap and all of the scores
corr_df = pd.concat(corr_df_list, axis=1)
### name the columns as overlap + column number
corr_df.columns = ['overlap_'+str(i) for i in range(corr_df.shape[1])]
### save as a csv file
#corr_df.to_csv('/home/delaram/sciFA/Results/simulation/metric_overlap_corr_df_sim'+str(num_sim_rounds)+'_March2024.csv')
#corr_df.to_csv('/home/delaram/sciFA/Results/simulation/metric_overlap_corr_df_sim'+str(num_sim_rounds)+'_April2024.csv')
corr_df.to_csv('/home/delaram/sciFA/Results/simulation/metric_overlap_corr_df_sim'+str(num_sim_rounds)+'_April2024_v2.csv')

### visaulize the results using visualize_simulation.R



bimodality_scores_df = pd.DataFrame(bimodality_scores).T
bimodality_scores_df.columns = bimodality_metrics
print(bimodality_scores_df.head())