import numpy as np
import scanpy as sc
import pandas as pd
import statsmodels as sm

def utils_test():
    print('sciRED utils is working')


def get_data_array(data) -> np.array:
    """Return the data matrix as a numpy array, and the number of cells and genes.
    data: AnnData object
    """

    data_numpy = data.X.toarray()

    ## working with the rat data
    num_cells = data_numpy.shape[0]
    num_genes = data_numpy.shape[1]

    genes = data.var_names

    print(num_cells, num_genes)

    return data_numpy, genes, num_cells, num_genes


def get_highly_variable_gene_indices(data_numpy, num_genes=2000, random=False):
    '''
    get the indices of the highly variable genes
    data_numpy: numpy array of the data (n_cells, n_genes)
    num_genes: number of genes to select
    random: whether to randomly select the genes or select the genes with highest variance
    '''
    if random:
        ### randomly select 1000 genes
        gene_idx = random.sample(range(0, data_numpy.shape[1]), num_genes)
    else:
        ### calculate the variance for each gene
        gene_vars = np.var(data_numpy, axis=0)
        ### select the top num_genes genes with the highest variance
        gene_idx = np.argsort(gene_vars)[-num_genes:]


    return gene_idx



def get_sub_data(data, num_genes=2000) -> tuple:    
    ''' subset the data matrix to the top num_genes genes
    y: numpy array of the gene expression matrix (n_cells, n_genes)
    random: whether to randomly select the genes or select the genes with highest variance
    num_genes: number of genes to select
    '''


    data_numpy = data.X.toarray()
    cell_sums = np.sum(data_numpy,axis=1) # row sums - library size
    gene_sums = np.sum(data_numpy,axis=0) # col sums - sum reads in a gene
    data = data[cell_sums!=0,gene_sums != 0] ## cells, genes

    data_numpy = data.X.toarray()
    ### calculate the variance for each gene
    gene_vars = np.var(data_numpy, axis=0)
    ### select the top num_genes genes with the highest variance
    gene_idx = np.argsort(gene_vars)[-num_genes:]

    #### select num_genes genes based on variance
    ## sort the gene_idx in ascending order
    gene_idx = np.sort(gene_idx)
    data = data[:,gene_idx]

    ### subset the data matrix to the top num_genes genes
    return data, gene_idx


def get_binary_covariate_v1(covariate, covariate_level, data) -> np.array:
    ''' return a binary covariate vector for a given covariate and covariate level
    covariate: a column of the dat object metadata
    covariate_level: one level of the covariate
    data: AnnData object
    '''
    covariate_list = np.zeros((data.obs.shape[0]))
    for i in range(data.obs.shape[0]):
        ### select the ith element of 
        if data.obs[[covariate]].squeeze()[i] == covariate_level:
            covariate_list[i] = 1
    return covariate_list


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


def get_design_mat(metadata_col, data) -> np.array:
    ''' return a onehot encoded design matrix for a given column of the dat object metadata
    a_metadata_col: a column of the dat object metadata
    data: AnnData object
    '''
    
    column_levels = data.obs[metadata_col].unique() 
    dict_covariate = {}
    for column_level in column_levels:
        print(column_level)
        dict_covariate[column_level] = get_binary_covariate(data.obs[[metadata_col]].squeeze(), column_level)

    #### stack colummns of dict_covariate 
    x = np.column_stack(([dict_covariate[column] for column in column_levels]))
    return x



def get_library_design_mat(data, lib_size='nCount_RNA'): # nCount_originalexp for scMixology
    ''' return a design matrix for the library size covariate - equivalent to performing normalization
    data: AnnData object
    lib_size: the library size covariate name in the AnnData object
    '''
    x = np.column_stack((np.ones(data.shape[0]), np.array(data.obs[lib_size])))
    return x


def get_scaled_vector(a_vector):
    ''' scale a vector to be between 0 and 1
    a_vector: a numpy array
    '''
    ### scale the vector to be between 0 and 1
    a_vector_scaled = (a_vector - np.min(a_vector))/(np.max(a_vector) - np.min(a_vector))
    return a_vector_scaled



