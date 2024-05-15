"""
simulate a factor with multiple covariates and calculate the overlap between the covariates
and the matching scores between the covariates and the factor
calculate the correlation between the overlap and the matching scores
repeat the simulation n times and calculate the average correlation between the overlap and all of the scores
save the results in a csv file
"""
import numpy as np
import pandas as pd
import time

from sciRED import ensembleFCA as efca
from sciRED import metrics as met
import simulation as sim

np.random.seed(10)

num_sim_rounds = 100
#num_sim_rounds = 2
num_factors = 20
num_mixtures = 2 ## each gaussian represents a covariate level 
num_samples = 1000

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
    fcat = efca.FCAT(covariate_vector, factor_scores)
    all_covariate_levels = fcat.index.values

    #################################### 
    #### calculate the overlap and matching scores for all the factors
    #################################### 
    match_score_mat_meanImp_list = []

    for i in range(num_factors): ## i is the factor index
        match_score_mat_meanImp_list.append(sim.get_pairwise_match_score_matrix(fcat,i))
    match_score_mat_meanImp_flat = sim.convert_matrix_list_to_vector(match_score_mat_meanImp_list)


    #### Bimodality scores
    silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores, wvrs = met.kmeans_bimodal_score(factor_scores, time_eff=False)
    bimodality_index = met.bimodality_index(factor_scores)
    dip_scores, pval_scores = met.get_dip_test_all(factor_scores)

    #### Effect size
    factor_variance = met.factor_variance(factor_scores)

    ## Specificity
    simpson_fcat = met.simpson_diversity_index(fcat)
    entropy_fcat = met.get_entropy(fcat)

    ### hemegeneity
    asv_arith = met.average_scaled_var(factor_scores, covariate_vector=covariate_vector, mean_type='arithmetic')
    asv_geo = met.average_scaled_var(factor_scores, covariate_vector=covariate_vector, mean_type='geometric')

    bimodality_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'wvrs',
                           'bimodality_index', 'dip_score']
    bimodality_scores = [ silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores, wvrs ,
                         bimodality_index, dip_scores]

    #### calculate the correlation between the overlap and all of the scores and save in a dataframe
    corr_df_temp = pd.DataFrame()
    for i in range(len(bimodality_metrics)):
        corr_df_temp[bimodality_metrics[i]] = [np.corrcoef(overlap_mat_flat, bimodality_scores[i])[0,1]]

    corr_df_temp['ASV_arith'] = [np.corrcoef(overlap_mat_flat, asv_arith)[0,1]]
    corr_df_temp['ASV_geo'] = [np.corrcoef(overlap_mat_flat, asv_geo)[0,1]]

    corr_df_temp['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance)[0,1]]

    ## only include in case #covariate levels > 3 - gini is a single value cant be saved in corr_df_temp
    corr_df_temp['entropy_fcat'] = [np.corrcoef(overlap_mat_flat, entropy_fcat)[0,1]]
    corr_df_temp['simpson_fcat'] = [np.corrcoef(overlap_mat_flat, simpson_fcat)[0,1]]

        
    corr_df_temp = corr_df_temp.T
    corr_df_temp.columns = ['overlap']
    #corr_df_temp = corr_df_temp.sort_values(by='overlap', ascending=False)
    corr_df_list.append(corr_df_temp)

    end_time_in = time.time()
    print('simulation: ', str(i), ' time: ', end_time_in - start_time_in)


end_time = time.time()
print('time: ', end_time - start_time)
time_minutes = round((end_time - start_time)/60, 2)
print('time in minutes: ', time_minutes)

### calculate the average correlation between the overlap and all of the scores
corr_df = pd.concat(corr_df_list, axis=1)
corr_df.columns = ['overlap_'+str(i) for i in range(corr_df.shape[1])]
corr_df.to_csv('/home/delaram/sciRED/simulation/metric_overlap_corr_df_sim'+str(num_sim_rounds)+'.csv')
