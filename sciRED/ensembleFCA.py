# training classifiers for feature importance on a classification problem
# matching pca factors to different covariates in the data

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

import skimage as ski
import scipy.stats as ss
import scipy as sp



def get_binary_covariate(covariate_vec, covariate_level) -> np.array:
    ''' return a binary covariate vector for a given covariate and covariate level
    covariate_vec: a vector of values for a covariate
    covariate_level: one level of the covariate
    '''
    covariate_list = np.zeros((len(covariate_vec)))
    for i in range(len(covariate_vec)):
        ### select the ith element of 
        if covariate_vec[i] == covariate_level:
            covariate_list[i] = 1
    return covariate_list



def get_AUC_alevel(a_factor, a_binary_cov) -> float:
    '''
    calculate the AUC of a factor for a covariate level
    return the AUC and the p-value of the U test
    a_factor: a factor score
    covariate_vector: a vector of the covariate
    covariate_level: a level of the covariate

    '''
    n1 = np.sum(a_binary_cov==1)
    n0 = len(a_factor)-n1
    
    ### U score manual calculation
    #order = np.argsort(a_factor)
    #rank = np.argsort(order)
    #rank += 1   
    #U1 = np.sum(rank[covariate_vector == covariate_level]) - n1*(n1+1)/2

    ### calculate the U score using scipy
    scipy_U = sp.stats.mannwhitneyu(a_factor[a_binary_cov == 1] , 
                                    a_factor[a_binary_cov != 1] , 
                                    alternative="two-sided", use_continuity=False)
    
    AUC1 = scipy_U.statistic/ (n1*n0)
    return AUC1, scipy_U.pvalue



def get_AUC_all_factors_alevel(factor_scores, a_binary_cov) -> list:
    '''
    calculate the AUC of all the factors for a covariate level
    return a list of AUCs for all the factors
    factor_scores: a matrix of factor scores
    a_binary_cov: a binnary vector of the covariate
    '''
    AUC_alevel_factors = []
    wilcoxon_pvalue_alevel_factors = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        AUC, wilcoxon_pvalue = get_AUC_alevel(a_factor, a_binary_cov)

        ### convert to same scale as feature importance
        AUC = abs((AUC - 0.5)*2)

        AUC_alevel_factors.append(AUC)
        wilcoxon_pvalue_alevel_factors.append(wilcoxon_pvalue)
    ### convert to numpy array
    AUC_alevel_factors = np.asarray(AUC_alevel_factors)
    return AUC_alevel_factors



def get_importance_df(factor_scores, a_binary_cov, time_eff=True) -> pd.DataFrame:
    '''
    calculate the importance of each factor for each covariate level
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    a_binary_cov: numpy array of the binary covariate for a covariate level (n_cells, )
    time_eff: if True, skip RandomForest which is time consuming
    force_all: if True, include KNeighbors_permute which often has lower performance 
    '''

    models = {'LogisticRegression': LogisticRegression(), 
              'DecisionTree': DecisionTreeClassifier(), 
              'RandomForest': RandomForestClassifier(), 
              'XGB': XGBClassifier(), 
              'KNeighbors_permute': KNeighborsClassifier()}
    
    if time_eff:
        ### remove RandomForest, KN from the models dictionary
        models.pop('RandomForest')
        models.pop('KNeighbors_permute')
        
    importance_dict = {}


    for model_name, model in models.items():
        X, y = factor_scores, a_binary_cov
        model.fit(X, y)

        if model_name == 'LogisticRegression':
            ### use the absolute value of the logistic reg coefficients as the importance - for consistency with other classifiers
            importance_dict[model_name] = np.abs(model.coef_)[0]
            #importance_dict[model_name] = model.coef_[0]

        elif model_name in ['DecisionTree', 'RandomForest', 'XGB']:
            # get importance values
            importance_dict[model_name] = model.feature_importances_

        elif model_name == 'KNeighbors_permute':
            # perform permutation importance
            results = permutation_importance(model, X, y, scoring='accuracy')
            importance_dict[model_name] = np.abs(results.importances_mean)
    
    #### adding AUC as a measure of importance
    AUC_alevel = get_AUC_all_factors_alevel(factor_scores, a_binary_cov)
    importance_dict['AUC'] = AUC_alevel

    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', 
                                           columns=['F'+str(i) for i in range(1, factor_scores.shape[1]+1)])
    return importance_df




def get_mean_importance_level(importance_df_a_level, scale, mean) -> np.array:
    ''' 
    calculate the mean importance of one level of a given covariate and returns a vector of length of number of factors
    importance_df_a_level: a dataframe of the importance of each factor for a given covariate level
    scale: 'standard', 'minmax' or 'rank', 'pearson'
    standard: scale each row of the importance_df_np to have zero mean and unit variance "REMOVED"
    minmax: scale each row of the importance_df_np to be between 0 and 1
    rank: replace each row of the importance_df_np with its rank
    mean: 'arithmatic' or 'geometric'
    arithmatic: calculate the arithmatic mean of each column
    geometric: calculate the geometric mean of each column
'
    '''
    importance_df_np = np.asarray(importance_df_a_level)
    ### normalize the importance score of each classifier in importance_df_np matrix
    if scale == 'standard':
        ### scale each row of the importance_df_np ( a model's importance results) to have zero mean and unit variance
        importance_df_np = (importance_df_np - importance_df_np.mean(axis=1, keepdims=True))/importance_df_np.std(axis=1, keepdims=True)

    if scale == 'minmax':
        ### scale each row of the importance_df_np to be between 0 and 1
        importance_df_np = (importance_df_np - importance_df_np.min(axis=1, keepdims=True))/(importance_df_np.max(axis=1, keepdims=True) - importance_df_np.min(axis=1, keepdims=True))
    
    if scale == 'rank':
        ### replace each row of the importance_df_np with its rank
        importance_df_np = np.apply_along_axis(ss.rankdata, 1, importance_df_np)
        ### for each row, devide ranks to n (number of factors) to get a value between 0 and 1
        importance_df_np = importance_df_np/importance_df_np.shape[1]

    ### calculate the mean of the importance_df_np matrix
    if mean == 'arithmatic':

        #### if any value in a column is equal or less than zero, add a small value to it - log of zero is undefined
        if np.any(importance_df_np == 0):
                importance_df_np[importance_df_np == 0] = 1e-10
        ### if any value is less than zero, replace with absolute value
        if np.any(importance_df_np < 0):
            importance_df_np[importance_df_np < 0] = np.abs(importance_df_np[importance_df_np < 0])
        ### calculate the arithmatic mean of each column
        importance_df = np.mean(importance_df_np, axis=0)

    if mean == 'geometric':
        
        #### if any value in a column is equal tozero, add a small value to it - log of zero is undefined
        if np.any(importance_df_np == 0):
                importance_df_np[importance_df_np == 0] = 1e-10
        ### if any value is less than zero, replace with absolute value
        if np.any(importance_df_np < 0):
            importance_df_np[importance_df_np < 0] = np.abs(importance_df_np[importance_df_np < 0])
        ### calculate the geometric mean of each column
        importance_df = ss.gmean(importance_df_np, axis=0)

    return importance_df



def FCAT(covariate_vec, factor_scores, 
         scale='standard', mean='arithmatic', time_eff=True) -> pd.DataFrame:
    '''
    calculate the mean importance of all levels of a given covariate and returns a dataframe of size (num_levels, num_components)
    covariate_vec: numpy array of the covariate vector (n_cells, )
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''

    mean_importance_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, factor_scores.shape[1]+1)])

    for covariate_level in np.unique(covariate_vec):
        print('covariate_level: ', covariate_level)

        a_binary_cov = get_binary_covariate(covariate_vec, covariate_level)
        importance_df_a_level = get_importance_df(factor_scores, a_binary_cov, time_eff=time_eff)
        mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale, mean)

        print('mean_importance_a_level:', mean_importance_a_level)
        mean_importance_df.loc[covariate_level] = mean_importance_a_level

    return mean_importance_df


def get_percent_matched_factors(mean_importance_df, threshold) -> float:
      total_num_factors = mean_importance_df.shape[1]
      matched_factor_dist = np.sum(mean_importance_df > threshold)

      num_matched_factors = np.sum(matched_factor_dist>0)
      percent_matched = np.round((num_matched_factors/total_num_factors)*100, 2)
      return matched_factor_dist, percent_matched


def get_percent_matched_covariates(mean_importance_df, threshold) -> float:
      total_num_covariates = mean_importance_df.shape[0]
      matched_covariate_dist = np.sum(mean_importance_df > threshold, axis=1)

      num_matched_cov = np.sum(matched_covariate_dist>0)
      percent_matched = np.round((num_matched_cov/total_num_covariates)*100, 2)
      return matched_covariate_dist, percent_matched



def get_otsu_threshold(values) -> float:
      '''
      This function calculates the otsu threshold of the feature importance scores
      :param values: a 1D array of values
      :return: threshold
      '''
      threshold = ski.filters.threshold_otsu(values)
      return threshold