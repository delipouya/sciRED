
import numpy as np
import pandas as pd
import statsmodels as sm
import scanpy as sc

def import_AnnData(path_to_file) -> sc.AnnData:
    """Import data from a file and return a numpy array.
    path_to_file: path to the file
    """
    #### import the immune subpopulation of the rat samples
    data = sc.read(path_to_file) ## attributes removed
    data.var_names_make_unique()

    ### renaming the meta info column names: https://github.com/theislab/scvelo/issues/255
    data.__dict__['_raw'].__dict__['_var'] = data.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

    return data


def get_metadata_scMix(data) -> tuple:
    """Return the metadata of the scMixology dataset, including cell-line, sample and protocol information.
    data: AnnData object
    """

    #### sample metadata
    y_cell_line = data.obs.cell_line_demuxlet
    y_sample = data.obs[['sample']].squeeze()

    #### adding a column to data object for protocol
    ## empty numpy array in length of the number of cells
    y_protocol = np.empty((data.n_obs), dtype="S10")

    for i in range(data.n_obs):
        if data.obs['sample'][i] in ['sc_10x', 'sc_10X']:
            y_protocol[i] = 'sc_10X'

        elif data.obs['sample'][i] == 'Dropseq':
            y_protocol[i] = 'Dropseq'
            
        else:
            y_protocol[i] = 'CELseq2'

    
    # data.obs['protocol'] = y_protocol
    y_protocol = pd.Series(y_protocol)
    y_protocol.unique()

    return y_cell_line, y_sample, y_protocol
    


def get_metadata_ratLiver(data) -> tuple:
    """Return the metadata of the healthy rat liver dataset, including sample, strain and cluster information.
    data: AnnData object
    """

    #### sample metadata
    y_cluster = data.obs.cluster.squeeze()
    y_sample = data.obs[['sample']].squeeze()
    y_strain = data.obs.strain.squeeze()
    y_cell_type = data.obs.annotation.squeeze()

    return y_sample, y_strain, y_cluster, y_cell_type


def get_metadata_humanLiver(data) -> tuple:
    """Return the metadata of the healthy human liver dataset, including sample, cell type information.
    data: AnnData object
    """
    y_sample = data.obs['sample'].squeeze()
    y_cell_type = data.obs['cell_type'].squeeze()
    return y_sample, y_cell_type



def get_metadata_humanKidney(data) -> tuple:
    """Return the metadata of the healthy human kidney dataset, including sex, sampleID, cell type information.
    data: AnnData object
    """
    y_sample = data.obs['sampleID'].squeeze()
    y_cell_type = data.obs['Cell_Types_Broad'].squeeze()
    y_cell_type_sub = data.obs['Cell_Types_Subclusters'].squeeze()
    y_sex = data.obs['sex'].squeeze()

    return y_sample, y_sex, y_cell_type, y_cell_type_sub 



def get_metadata_humanPBMC(data) -> tuple:
    """Return the metadata of the stimulated human pbmc dataset, including sample, stimulation, cluster and cell type information.
    data: AnnData object
    """
    y_sample = data.obs['ind'].squeeze()
    y_stim = data.obs['stim'].squeeze()
    y_cell_type = data.obs['cell'].squeeze()
    y_cluster = data.obs['cluster'].squeeze()

    return y_sample, y_stim, y_cell_type, y_cluster 

