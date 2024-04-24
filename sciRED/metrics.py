import numpy as np
import pandas as pd
import scipy as sp

import functions_processing as fproc

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import diptest


def get_factor_simpson_D_index(x):
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



def get_all_factors_simpson_D_index(mean_importance_df) -> list:
    '''
    calculate the simpson index of all the factors based on the mean importance matrix
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    '''
    factor_simpson_D_all = []
    for factor_i in range(mean_importance_df.shape[1]):
        ### get the importance of the factor for each covariate level
        x = mean_importance_df.iloc[:, factor_i]
        
        simpson_D_index = get_factor_simpson_D_index(x)
        factor_simpson_D_all.append(simpson_D_index)
    return factor_simpson_D_all



def get_all_factors_simpson(mean_importance_df) -> list:
    '''
    calculate the simpson index of all the factors based on the mean importance matrix
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    '''
    factor_simpson_all = []
    for factor_i in range(mean_importance_df.shape[1]):
        ### get the importance of the factor for each covariate level
        x = mean_importance_df.iloc[:, factor_i]
        
        simpson_index = 1 - get_factor_simpson_D_index(x)
        factor_simpson_all.append(simpson_index)
    return factor_simpson_all

#define function to calculate Gini coefficient
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



def get_all_factors_gini(mean_importance_df) -> float:
    '''
    calculate the gini score for the mean importance matrix
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    '''
    
    ### convert the mean_importance_df to a one dimensional numpy array
    ### apply gini to each row of the mean_importance_df (each covariate level) and 
    ## then take the mean for the whole matrix    
    gini = mean_importance_df.apply(get_gini, axis=0).mean()
    return gini



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



def get_factor_entropy_all(mean_importance_df) -> list:
    '''
    calculate the entropy of all the factors based on the mean importance matrix
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    '''
    H_all = []
    for factor_i in range(mean_importance_df.shape[1]):
        ### get the importance of the factor for each covariate level
        H = get_factor_entropy(mean_importance_df.iloc[:, factor_i])
        H_all.append(H)
    return H_all



def get_factor_variance_all(factor_scores) -> list:
    ''' calculate the variance of all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    factor_variance_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        factor_variance = np.var(a_factor)
        factor_variance_all.append(factor_variance)
    return factor_variance_all


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


def get_a_factor_ASV(a_factor, covariate_vector, mean_type='geometric') -> float:
    '''
    calculate an average for the relative scaled variance for all the levels in a covariate
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    mean_type: the type of mean to calculate the average scaled variance
    '''

    ### calculate the relative scaled variance for all the levels in a covariate
    scaled_variance_all = get_SV_all_levels(a_factor, covariate_vector)
    print('mean type: ', mean_type)
    ### calculate the geometric mean of the scaled variance for all levels of the covariate
    if mean_type == 'geometric':
        #RSV = np.exp(np.mean(np.log(scaled_variance_all))) 
        #RSV = np.exp(np.log(scaled_variance_all).mean())
        ### replace the zero values with a small number
        scaled_variance_all[scaled_variance_all == 0] = 1e-10
        ### calculate the geometric mean using the scipy gmean function
        RSV = sp.stats.gmean(scaled_variance_all)


    if mean_type == 'arithmetic':
        # calculate the arithmetic mean using the numpy mean function
        RSV = np.mean(scaled_variance_all)

    return RSV


def get_ASV_all(factor_scores, covariate_vector, mean_type='geometric') -> list:
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


def get_factors_SV_all_levels(factor_scores, covariate_vector) -> np.array:
    '''
    calculate the scaled variance for all the levels in a covariate for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    '''
    SV_all_factors = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        SV_all = get_SV_all_levels(a_factor, covariate_vector)
        
        SV_all_factors.append(SV_all)
    SV_all_factors = np.asarray(SV_all_factors)
    return SV_all_factors.T ### traspose the matrix to have the factors in columns and cov levels in rows




def get_total_sum_of_squares(a_factor) -> float:
    '''
    calculate the total sum of squares for a factor
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    '''
    a_factor = a_factor.reshape(-1,1)
    tss = np.sum((a_factor - np.mean(a_factor))**2)
    return tss

### calculate within cluster sum of squares for each factor based on clustering labels
def get_factor_wcss(a_factor, labels) -> list:
    '''
    calculate the sum of squares error for each factor based on the clustering labels
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    labels: numpy array of the clustering labels for all the cells (n_cells, 1)
    '''
    a_factor = a_factor.reshape(-1,1)
    sse = np.sum((a_factor - np.mean(a_factor[labels==0]))**2) + np.sum((a_factor - np.mean(a_factor[labels==1]))**2)
        
    return sse

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

def get_variance_reduction_score(a_factor, labels) -> float:
    '''
    calculate the variance reduction score for a factor based on the clustering labels
    VRS measures the proportion of variance reduction when splitting the data into two clusters. 
    The value of this score lies in the interval [0; 1], and a low score indicates an informative split.
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    labels: numpy array of the clustering labels for all the cells (n_cells, 1)
    '''
    tss = get_total_sum_of_squares(a_factor)
    sse = get_factor_wcss(a_factor, labels)
    #vrs = 1 - sse/tss
    vrs = sse/tss
    return vrs

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



def get_kmeans_scores(factor_scores, num_groups=2, time_eff=True) -> list:
    '''
    fit a kmeans model to each factor and calculate the and silhouette scores for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    num_groups: number of groups to fit the kmeans model

    '''
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    #wvrs = []

    for i in range(factor_scores.shape[1]):
        kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(factor_scores[:,i].reshape(-1,1))
        labels = kmeans.labels_

        silhouette_scores.append(silhouette_score(factor_scores[:,i].reshape(-1,1), labels))
        

        if not time_eff:
            calinski_harabasz_scores.append(calinski_harabasz_score(factor_scores[:,i].reshape(-1,1), labels))
            davies_bouldin_scores.append(davies_bouldin_score(factor_scores[:,i].reshape(-1,1), labels))
            #wvrs.append(get_weighted_variance_reduction_score(factor_scores[:,i].reshape(-1,1), labels))

            ## reverse davies_bouldin_scores in a way that lower values indicate better-defined clusters (reverse)
            davies_bouldin_scores = [1/x for x in davies_bouldin_scores]
            ### scale davies_bouldin_scores between 0 and 1
            davies_bouldin_scores = fproc.get_scaled_vector(davies_bouldin_scores)

    if not time_eff:    
        return silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores
    
    return silhouette_scores


def get_gmm_scores(factor_scores, num_groups=2, time_eff=True) -> list:
    '''
    fit a gaussian mixture model to each factor and calculate the silhouette scores for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    num_groups: number of groups to fit the gaussian mixture model
    '''
    silhouette_scores = []
    vrs = []
    wvrs = []
    means = []
    cov_list = []
    weights = []
    
    for i in range(factor_scores.shape[1]):
        gmm = GaussianMixture(n_components=num_groups, covariance_type='full', random_state=0)
        gmm.fit(factor_scores[:,i].reshape(-1,1))

        ### save the mean, coc and weight of gmm in lists
        means.append(gmm.means_)
        cov_list.append(gmm.covariances_)
        weights.append(gmm.weights_)

        labels = gmm.predict(factor_scores[:,i].reshape(-1,1))

        if not time_eff:
            #aic_scores.append(gmm.aic(factor_scores[:,i].reshape(-1,1)))
            vrs.append(get_variance_reduction_score(factor_scores[:,i].reshape(-1,1), labels))
            wvrs.append(get_weighted_variance_reduction_score(factor_scores[:,i].reshape(-1,1), labels))
            
            
        silhouette_scores.append(silhouette_score(factor_scores[:,i].reshape(-1,1), labels))

    if not time_eff:
        return silhouette_scores, vrs, wvrs #, means, cov_list, weights
    return silhouette_scores
        

    



def get_likelihood_ratio_test(a_factor, num_groups=2) -> float:
    '''
    calculate the likelihood ratio test for a factor
    likelihood ratio of a normal model and a mixture normal model to identify bimodal distributions was suggested by Ertel and Tozeren 
    Small ratios indicate that the distribution is unimodal, whereas large ratios suggest that the expression values are bimodally distributed.
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    num_groups: number of groups to fit the gaussian mixture model
    '''
    gmm = GaussianMixture(n_components=num_groups, covariance_type='full', random_state=0)
    gmm.fit(a_factor.reshape(-1,1))
    labels = gmm.predict(a_factor.reshape(-1,1))
    ### calculate the likelihood of the gaussian model 
    log_likelihood_normal = np.sum(gmm.score_samples(a_factor.reshape(-1,1)))

    ### calculate the likelihood of the mixture normal model

    gmm_predict = gmm.predict_proba(a_factor.reshape(-1,1))
    
    ### if the gmm_predict is None, set the log_likelihood_mixture to zero
    if gmm_predict is None:
        log_likelihood_mixture = 0
    ### if any element is equal to zero add a small number to it to avoid log(0)
    elif np.any(gmm_predict == 0):
        gmm_predict[gmm_predict == 0] = 1e-10
        log_likelihood_mixture = np.sum(np.log(gmm_predict))

    else:
        log_likelihood_mixture = np.sum(np.log(gmm_predict))

    ### calculate the likelihood ratio test
    #lrt = -2*(log_likelihood_normal - log_likelihood_mixture)
    lrt = log_likelihood_normal - log_likelihood_mixture
    
    return lrt


def get_likelihood_ratio_test_all(factor_scores, num_groups=2) -> list:
    '''
    calculate the likelihood ratio test for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    num_groups: number of groups to fit the gaussian mixture model
    '''
    lrt_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        lrt = get_likelihood_ratio_test(a_factor, num_groups)
        lrt_all.append(lrt)
    return lrt_all


def get_bimodality_index(a_factor, num_groups=2) -> float:
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
    
    
def get_bimodality_index_all(factor_scores, num_groups=2) -> list:
    '''
    calculate the bimodality index for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    num_groups: number of groups to fit the gaussian mixture model
    '''
    bi_index_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        bi_index = get_bimodality_index(a_factor, num_groups)
        bi_index_all.append(float(bi_index))
    return bi_index_all



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



def get_factor_kurtosis(a_factor) -> float:
    '''
    calculate the kurtosis of a factor
    A gaussian distribution has kurtosis K = 0, whereas most non-gaussian distributions have either K > 0 or K < 0. 
    Specifically, a mixture of two approximately equal mass normal distributions must have negative kurtosis since 
    the two modes on either side of the center of mass effec- tively flatten out the distribution. 
    A mixture of two normal distributions with highly unequal mass must have positive kurtosis since 
    the smaller distribution lengthens the tail of the more dominant normal distribution. 
    If there is an 80%-20% split of the samples into two groups, 
    then the kurtosis is close to 0. Therefore biologically interesting genes might be missed.
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    '''
    kurtosis = sp.stats.kurtosis(a_factor)
    return kurtosis



def get_factor_kurtosis_all(factor_scores) -> list:
    '''
    calculate the kurtosis of all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    kurtosis_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        kurtosis = get_factor_kurtosis(a_factor)
        kurtosis_all.append(kurtosis)
    return kurtosis_all


def median_absolute_deviation(a_factor) -> float:
    '''
    calculate the median absolute deviation of a factor
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    '''
    mad = np.median(np.abs(a_factor - np.median(a_factor)))
    return mad


def get_interquartile_range(a_factor) -> float:
    '''
    calculate the interquartile range of a factor
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    '''
    iqr = np.percentile(a_factor, 75) - np.percentile(a_factor, 25)
    return iqr


def get_outlier_sum_statistic(a_factor) -> float:
    '''
    calculate the outlier sum statistic of a factor
    Wi is large if there are many outliers in the data or few extreme outliers with high values, 
    and Wi is zero if there are no outliers. 
    a_factor: numpy array of the factor scores for all the cells (n_cells, 1)
    '''
    mad = median_absolute_deviation(a_factor)
    x_p = (a_factor - np.median(a_factor))/mad
    
    q_25 = np.percentile(a_factor, 25)
    q_75 = np.percentile(a_factor, 75)
    iqr = get_interquartile_range(a_factor)

    # # W =∑x′ ⋅I[x′ >q (i)+IQR(i)]
    W_pos = np.sum(x_p[x_p> (q_75+iqr)])
    W_neg = np.sum(x_p[x_p< (q_25-iqr)])
    ### abs max of W_pos and W_neg
    W = np.max([np.abs(W_pos), np.abs(W_neg)])
    
    return W


def get_outlier_sum_statistic_all(factor_scores) -> list:
    '''
    calculate the outlier sum statistic of all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    W_all = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        W = get_outlier_sum_statistic(a_factor)
        W_all.append(W)
    return W_all



def get_scaled_metrics(all_metrics_df) -> np.array:
    '''
    Return numpy array of the scaled all_metrics pandas df based on each metric 
    all_metrics_df: a pandas dataframe of the metrics for all the factors
    '''
    all_metrics_np = all_metrics_df.to_numpy()
    
    ### scale the metrics in a loop
    all_metrics_scaled = np.zeros(all_metrics_np.shape)
    for i in range(all_metrics_np.shape[1]):
        all_metrics_scaled[:,i] = fproc.get_scaled_vector(all_metrics_np[:,i])


    return all_metrics_scaled



def get_AUC_alevel(a_factor, covariate_vector, covariate_level) -> float:
    '''
    calculate the AUC of a factor for a covariate level
    return the AUC and the p-value of the U test
    a_factor: a factor score
    covariate_vector: a vector of the covariate
    covariate_level: a level of the covariate

    '''
    n1 = np.sum(covariate_vector==covariate_level)
    n0 = len(a_factor)-n1
    
    ### U score manual calculation
    #order = np.argsort(a_factor)
    #rank = np.argsort(order)
    #rank += 1   
    #U1 = np.sum(rank[covariate_vector == covariate_level]) - n1*(n1+1)/2

    ### calculate the U score using scipy
    scipy_U = sp.stats.mannwhitneyu(a_factor[covariate_vector == covariate_level] , 
                                    a_factor[covariate_vector != covariate_level] , 
                                    alternative="two-sided", use_continuity=False)
    
    AUC1 = scipy_U.statistic/ (n1*n0)
    return AUC1, scipy_U.pvalue


def get_AUC_all_levels(a_factor, covariate_vector) -> list:
    '''
    calculate the AUC of a factor for all the covariate levels
    return a list of AUCs for all the covariate levels
    a_factor: a factor score
    covariate_vector: a vector of the covariate
    '''
    AUC_all = []
    wilcoxon_pvalue_all = []
    for covariate_level in np.unique(covariate_vector):
        AUC1, pvalue = get_AUC_alevel(a_factor, covariate_vector, covariate_level)
        ### AUC: 0.5 is random, 1 is perfect 
        ### to convert to feature importance, subtract 0.5 and multiply by 2
        AUC1_scaled = np.abs((AUC1*2)-1)
        AUC_all.append(AUC1_scaled)
        ### convert the pvalue to a -log10 scale to handle the exponential distribution. High values are better? #TODO: check this - remove the negative sign??
        wilcoxon_pvalue_all.append(-np.log10(pvalue))

    ### scale the wilcoxon pvalues to be between 0 and 1
    wilcoxon_pvalue_all = fproc.get_scaled_vector(wilcoxon_pvalue_all)
    ## reverse the direction of the scaled pvalues
    wilcoxon_pvalue_all = 1 - wilcoxon_pvalue_all
    return AUC_all, wilcoxon_pvalue_all




def get_AUC_all_factors(factor_scores, covariate_vector) -> list:
    '''
    calculate the AUC of all the factors for all the covariate levels
    return a list of AUCs for all the factors
    factor_scores: a matrix of factor scores
    covariate_vector: a vector of the covariate
    '''
    AUC_all_factors = []
    wilcoxon_pvalue_all_factors = []
    for i in range(factor_scores.shape[1]):
        a_factor = factor_scores[:,i]
        AUC_all, wilcoxon_pvalue_all = get_AUC_all_levels(a_factor, covariate_vector)
        AUC_all_factors.append(AUC_all)
        wilcoxon_pvalue_all_factors.append(wilcoxon_pvalue_all)
    return AUC_all_factors, wilcoxon_pvalue_all_factors



def get_AUC_all_factors_df(factor_scores, covariate_vector) -> pd.DataFrame:
    '''
    calculate the AUC of all the factors for all the covariate levels
    return a dataframe of AUCs for all the factors
    factor_scores: a matrix of factor scores
    covariate_vector: a vector of the covariate
    '''
    AUC_all_factors, wilcoxon_pvalue_all_factors = get_AUC_all_factors(factor_scores, covariate_vector)

    AUC_all_factors_df = pd.DataFrame(AUC_all_factors).T
    AUC_all_factors_df.columns = ['F'+str(i+1) for i in range(factor_scores.shape[1])]
    AUC_all_factors_df.index = np.unique(covariate_vector)

    #wilcoxon_pvalue_all_factors_df = pd.DataFrame(wilcoxon_pvalue_all_factors).T
    #wilcoxon_pvalue_all_factors_df.columns = ['F'+str(i+1) for i in range(factor_scores.shape[1])]
    #wilcoxon_pvalue_all_factors_df.index = np.unique(covariate_vector)


    return AUC_all_factors_df



def get_reversed_AUC_df(AUC_all_factors_df) -> pd.DataFrame:
    '''
    calculate the 1 - the AUC of all the factors for all the covariate levels as a measure of homogeneous factors
    higher values (closer to 1) are more homogeneous
    return a dataframe of 1-AUCs for all the factors
    AUC_all_factors_df: a dataframe of AUCs for all the factors
    '''
    AUC_all_factors_df_1 = 1-AUC_all_factors_df
    return AUC_all_factors_df_1


def get_factor_binned(factor_scores, num_bins=10):
    '''
    bin a factor into 10 bins
    '''
    factor_scores_binned = np.zeros(factor_scores.shape)
    for i in range(num_bins):
        factor_scores_binned[np.where(factor_scores >= np.percentile(factor_scores, i*10))[0]] = i
    return factor_scores_binned


