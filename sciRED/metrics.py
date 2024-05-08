
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import scipy as sp

from sciRED.utils import preprocess as proc

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import diptest

def kmeans_bimodal_score(factor_scores, num_groups=2, time_eff=True) -> list:
    '''
    fit a kmeans model to each factor and calculate the and silhouette scores for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    num_groups: number of groups to fit the kmeans model

    '''
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    wvrs = []

    for i in range(factor_scores.shape[1]):
        kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(factor_scores[:,i].reshape(-1,1))
        labels = kmeans.labels_

        silhouette_scores.append(silhouette_score(factor_scores[:,i].reshape(-1,1), labels))
        

        if not time_eff:
            calinski_harabasz_scores.append(calinski_harabasz_score(factor_scores[:,i].reshape(-1,1), labels))
            davies_bouldin_scores.append(davies_bouldin_score(factor_scores[:,i].reshape(-1,1), labels))
            wvrs.append(get_weighted_variance_reduction_score(factor_scores[:,i].reshape(-1,1), labels))

            

    if not time_eff:   
        ## reverse davies_bouldin_scores in a way that lower values indicate better-defined clusters (reverse)
        davies_bouldin_scores = [1/x for x in davies_bouldin_scores]
        ### scale davies_bouldin_scores between 0 and 1
        davies_bouldin_scores = proc.get_scaled_vector(davies_bouldin_scores) 

        return silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores, wvrs
    
    return silhouette_scores



def get_factor_bimodality_index(a_factor, num_groups=2) -> float:
    '''
    calculate the bimodality index for a factor
    distribution of a factor with bimodal values can be expressed as a mixture of two normal distributions with means μ1 and μ2 
    and equal standard deviation. The standardized distance δ between the two populations is given by
    '''
    gmm = GaussianMixture(n_components=num_groups, covariance_type='full', random_state=0)
    gmm.fit(a_factor.reshape(-1,1))
    ### calculate the standard distance (sigma) by subtracting the means and dividing by the standard deviation
    sigma = np.abs(gmm.means_[0] - gmm.means_[1])/np.sqrt(gmm.covariances_[0])
    ### to identify bimodal factors, H0: sigma = 0, H1: sigma > 0
    ### calculate the bimodality index
    ### proportion of cells in the first component
    pi = np.sum(gmm.weights_[0])/np.sum(gmm.weights_)
    bi_index = np.sqrt(pi*(1-pi))*sigma
    return bi_index
    
    
def bimodality_index(factor_scores, num_groups=2) -> list:
    '''
    calculate the bimodality index for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    num_groups: number of groups to fit the gaussian mixture model
    '''
    bi_index_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        bi_index = get_factor_bimodality_index(a_factor, num_groups)
        bi_index_all.append(float(bi_index))
    return bi_index_all



def factor_variance(factor_scores) -> list:
    ''' calculate the variance of all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    factor_variance_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        factor_variance = np.var(a_factor)
        factor_variance_all.append(factor_variance)
    return factor_variance_all



def get_factor_simpson_diversity_index(x):
    '''
    calculate the simpson index of a factor - "a single factor" specificity based on feature importance scores
    x: numpy array of the factor match scores based on feature importance
    simpson index = 1 - simpson diversity index
    simpson diversity index = sum of the square of the probability of a factor being selected
    '''
    ### calculate the probability of a level being selected
    p_factor = x/np.sum(x)
    ### calculate the simpson diversity index
    simpson_diversity_index = np.sum(p_factor**2)
    
    return simpson_diversity_index



def simpson_diversity_index(mean_importance_df) -> list:
    '''
    calculate the simpson index of all the factors based on the mean importance matrix
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    '''
    factor_simpson_D_all = []
    for factor_i in range(mean_importance_df.shape[1]):
        ### get the importance of the factor for each covariate level
        x = mean_importance_df.iloc[:, factor_i]
        
        simpson_D_index = get_factor_simpson_diversity_index(x)
        factor_simpson_D_all.append(simpson_D_index)
    return factor_simpson_D_all



def get_scaled_variance_level(a_factor, covariate_vector, covariate_level) -> float:
    ''' 
    calculate the scaled variance of one factor and one covariate
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    covariate_level: one level of interest in the covariate
    '''
    ### select the cells in a_factor that belong to the covariate level
    a_factor_subset = a_factor[covariate_vector == covariate_level] 
    ### scaled variance of a factor and a covariate level
    scaled_variance = np.var(a_factor_subset)/np.var(a_factor) 
    return scaled_variance


def get_SV_all_levels(a_factor, covariate_vector) -> list:
    '''
    calculate the scaled variance for all the levels in a covariate
    represents how well mixed the factor is across each level of the covariate. 
    scaled_variance = 1 is well mixed, 0 is not well mixed
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    '''
    scaled_variance_all = []
    for covariate_level in covariate_vector.unique():
        scaled_variance = get_scaled_variance_level(a_factor, covariate_vector, covariate_level)
        scaled_variance_all.append(scaled_variance)
        
    return scaled_variance_all


def get_a_factor_ASV(a_factor, covariate_vector, mean_type='arithmetic') -> float:
    '''
    calculate an average for the relative scaled variance for all the levels in a covariate
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    mean_type: the type of mean to calculate the average scaled variance
    '''

    ### calculate the relative scaled variance for all the levels in a covariate
    scaled_variance_all = get_SV_all_levels(a_factor, covariate_vector)
    ### calculate the geometric mean of the scaled variance for all levels of the covariate
    if mean_type == 'geometric':
        ### replace the zero values with a small number
        scaled_variance_all[scaled_variance_all == 0] = 1e-10
        ### calculate the geometric mean using the scipy gmean function
        RSV = sp.stats.gmean(scaled_variance_all)


    if mean_type == 'arithmetic':
        # calculate the arithmetic mean using the numpy mean function
        RSV = np.mean(scaled_variance_all)

    return RSV


def average_scaled_var(factor_scores, covariate_vector, mean_type='arithmetic') -> list:
    '''
    calculate the average scaled variance for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    mean_type: the type of mean to calculate the average scaled variance
    '''

    ASV_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        ASV = get_a_factor_ASV(a_factor, covariate_vector, mean_type)
        ASV_all.append(ASV)
    return ASV_all



def get_factor_entropy(x) -> float:
    '''
    calculate the entropy of a factor 
    zero entropy means the factor is only macthed with one covariate. entropy=1 means the factor is matched with all the covariates equally
    entropy=0 means high specificity of the factor

    rev_entropy = 1 - entropy: higher rev_entropy means higher specificity
    a_factor: numpy array of the factor values
    '''
    ### caculate number of zeros in pk
    # num_zeros = np.count_nonzero(a_factor == 0)
    
    p_factor = x/np.sum(x)
    ### check if any element is zero
    if np.any(p_factor == 0):
        ### replace the zero values with a small number
        p_factor[p_factor == 0] = 1e-10
    ### calculate the entropy
    H = -sum(p_factor * np.log(p_factor))
    
    return H


def get_entropy(fcat) -> list:
    '''
    calculate the entropy of all the factors based on the mean importance matrix
    fcat: dataframe of mean importance of each factor for each covariate level
    '''
    H_all = []
    for factor_i in range(fcat.shape[1]):
        ### get the importance of the factor for each covariate level
        H = get_factor_entropy(fcat.iloc[:, factor_i])
        H_all.append(H)
    return H_all


def get_gini(x):
    '''
    calculate the gini score of a vector
    x: numpy array of the all factor match scores based on feature importance
    gini=0 means perfect equality - all the covariate levels are matched equally with the factor - not specific
    gini=1 means perfect inequality - the factor is only matched with one covariate level - very specific
    '''
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))



def fcat_gini(fcat) -> float:
    '''
    calculate the gini score for the mean importance matrix
    fcat: dataframe of mean importance of each factor for each covariate level
    '''
    
    ### convert the mean_importance_df to a one dimensional numpy array
    ### apply gini to each row of the mean_importance_df (each covariate level) and 
    ## then take the mean for the whole matrix    
    gini = fcat.apply(get_gini, axis=0).mean()
    return gini



def get_dip_test_all(factor_scores) -> list:
    '''
    This function calculates the Hartigan's dip test for unimodality for each factor individually.
    The dip statistic is defined as the maximum difference between an empirical distribution function and 
    the unimodal distribution function that minimizes that maximum difference.
    factor_scores: numpy array of shape (num_cells, num_factors)
    '''
    dip_scores = []
    pval_scores = []
    for i in range(factor_scores.shape[1]):
        dip, pval = diptest.diptest(factor_scores[:,i])
        dip_scores.append(dip)
        pval_scores.append(pval)

    return dip_scores, pval_scores



def get_total_sum_of_squares(a_factor) -> float:
    '''
    calculate the total sum of squares for a factor
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    '''
    a_factor = a_factor.reshape(-1,1)
    tss = np.sum((a_factor - np.mean(a_factor))**2)
    return tss


def get_factor_wcss_weighted(a_factor, labels) -> list:
    '''
    calculate the sum of squares error for each factor based on the clustering labels
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    labels: numpy array of the clustering labels for all the cells (n_cells, 1)
    '''
    a_factor = a_factor.reshape(-1,1)
    n0 = a_factor[labels==0].shape[0]
    n1 = a_factor[labels==1].shape[0]
    sse = (1/n0)*(np.sum((a_factor - np.mean(a_factor[labels==0]))**2)) + (1/n1)*(np.sum((a_factor - np.mean(a_factor[labels==1]))**2))
        
    return sse


def get_weighted_variance_reduction_score(a_factor, labels) -> float:
    '''
    calculate the weighted variance reduction score for a factor based on the clustering labels
    The weighted variance reduction score (WVRS) measures the variance reduction independent of the cluster sizes.
    In the numerator we calculate the mean of the two within cluster variances. 
    The value of this score can be larger than 1. Low score reflects bimodality. 
    WVRS has the ability to also identify splits into two clusters with extremely unequal sample sizes.
    '''
    tss = get_total_sum_of_squares(a_factor)
    sse_weighted = get_factor_wcss_weighted(a_factor, labels)
    n = a_factor.shape[0]
    wvrs = (n*sse_weighted)/(2*tss)
    return wvrs


def get_scaled_metrics(all_metrics_df) -> np.array:
    '''
    Return numpy array of the scaled all_metrics pandas df based on each metric 
    all_metrics_df: a pandas dataframe of the metrics for all the factors
    '''
    all_metrics_np = all_metrics_df.to_numpy()
    
    ### scale the metrics in a loop
    all_metrics_scaled = np.zeros(all_metrics_np.shape)
    for i in range(all_metrics_np.shape[1]):
        all_metrics_scaled[:,i] = proc.get_scaled_vector(all_metrics_np[:,i])

    return all_metrics_scaled



def FIST(all_metrics_dict) -> pd.DataFrame:
    '''
    calculate the FIST score for each factor based on the metrics
    all_metrics_dict: dictionary of all the metrics for each factor
    '''
    fist = pd.DataFrame(all_metrics_dict)
    fist_scaled = get_scaled_metrics(fist)

     ### remove numbers from heatmap cells
    fist_scaled_df = pd.DataFrame(fist_scaled)
    fist_scaled_df.columns = list(all_metrics_dict.keys())

    return fist_scaled_df


