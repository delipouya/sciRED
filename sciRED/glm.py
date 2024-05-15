import numpy as np
import statsmodels as sm
import time

def poissonGLM(y, x) -> dict:

    '''
    fit a Poisson GLM model to each gene of the data
    y: gene expression matrix, cells x genes
    x: design matrix
    num_vars: number of variables in the design matrix
    '''
    num_cells = y.shape[0]
    num_genes = y.shape[1]

    resid_pearson = []
    resid_deviance = []
    resid_response = []

    ### time the fitting process
    start_time = time.time()

    for i in range(len(y[0])):
        y_a_gene = y[:, i]
        model = sm.GLM(y_a_gene, x, family=sm.families.Poisson())
        result = model.fit()
        resid_pearson.append([result.resid_pearson])
        resid_deviance.append([result.resid_deviance])
        resid_response.append([result.resid_response])

    end_time = time.time()
    print('time to fit the model: ', end_time - start_time)

    resid_pearson = np.asarray(resid_pearson).reshape(num_genes, num_cells)
    resid_deviance = np.asarray(resid_deviance).reshape(num_genes, num_cells)
    resid_response = np.asarray(resid_response).reshape(num_genes, num_cells)

    glm_fit_dict = {'resid_pearson': resid_pearson, 
                    'resid_deviance': resid_deviance, 
                    'resid_response': resid_response}

    return glm_fit_dict

