import seaborn as sns
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import random
random.seed(0)

### make a legend patch given a numpy array of covariates and their colors for each sample 
def get_legend_patch(y_sample, sample_color):
    '''
    generate a legend patch for each sample in y_sample
    y_sample: the sample for each cell
    sample_color: the color for each sample
    '''

    ### make a dictionary of colors annd samples
    my_color = {y_sample[i]: sample_color[i] for i in range(len(y_sample))}

    ### make a legend patch based on my_color
    legend_patch = [mpatches.Patch(color=my_color[i], label=i) for i in np.unique(y_sample)]
    return legend_patch



def get_colors_dict_scMix(y_protocol, y_cell_line):
    '''
    generate a dictionary of colors for each cell in the scMix dataset
    y_protocol: the protocol for each cell
    y_cell_line: the cell line for each cell
    '''

    ### generating the list of colors for samples
    my_color = {b'sc_10X': 'palegreen', b'CELseq2':'yellow', b'Dropseq':'pink'}
    ### generate a list containing the corresponding color for each sample
    protocol_color = [my_color[y_protocol[i]] for i in range(len(y_protocol))]

    my_color = {'HCC827': 'springgreen', 'H1975':'red', 'H2228':'orchid'}
    cell_line_color = [my_color[y_cell_line[i]] for i in range(len(y_cell_line))]

    return {'protocol': protocol_color, 'cell_line': cell_line_color}



def get_colors_dict_ratLiver(y_sample, y_strain,y_cell_type):
    '''
    generate a dictionary of colors for each cell in the rat liver dataset
    y_sample: the sample for each cell
    y_strain: the strain for each cell
    y_cluster: the cluster for each cell
    '''

    my_color = {'DA_01': 'red','DA_02': 'orange', 'LEW_01': 'blue', 'LEW_02': 'purple'}
    sample_color = [my_color[y_sample[i]] for i in range(len(y_sample))]

    ### make a dictionary of colors for each strain in y_strain
    my_color = {'DA': 'red', 'LEW': 'blue'}
    strain_color = [my_color[y_strain[i]] for i in range(len(y_strain))]


    ### make a dictionary of colors for each 16 cluster in y_cluster. use np.unique(y_cell_type)
    ### generate 16 colors using the following code:
    my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_cell_type)}
    cell_type_color = [my_color[y_cell_type[i]] for i in range(len(y_cell_type))]

    return {'sample': sample_color, 'strain': strain_color, 'cell_type':cell_type_color}



def get_colors_dict_humanLiver(y_sample, y_cell_type):
    '''
    generate a dictionary of colors for each cell in the rat liver dataset
    y_sample: the sample for each cell
    y_strain: the strain for each cell
    y_cluster: the cluster for each cell
    '''

    ### make a dictionary of colors for each sample in y_sample
    my_color = {'P1TLH': 'red','P3TLH': 'orange', 'P2TLH': 'blue', 'P5TLH': 'purple','P4TLH': 'green'}
    sample_color = [my_color[y_sample[i]] for i in range(len(y_sample))]

    ### make a dictionary of colors for each 16 cluster in y_cluster. use np.unique(y_cluster)
    ### generate 16 colors using the following code:
    my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_cell_type)}
    cell_type_color = [my_color[y_cell_type[i]] for i in range(len(y_cell_type))]

    return {'sample': sample_color, 'cell_type':cell_type_color}


### ToDo: needs to be refined
def get_colors_dict_humanKidney(y_sample, y_sex, y_cell_type):
    '''
    generate a dictionary of colors for each cell in the rat liver dataset
    y_sample: the sample for each cell
    y_sex: the sex info for each cell
    y_cell_type: the cluster for each cell
    '''

    ### make a dictionary of colors for each sample in y_sample
    my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_sample)}
    sample_color = [my_color[y_sample[i]] for i in range(len(y_sample))]

    ### make a dictionary of colors for each strain in y_strain
    my_color = {'Male': 'forestgreen', 'Female': 'hotpink'}
    sex_color = [my_color[y_sex[i]] for i in range(len(y_sex))]

    ### make a dictionary of colors for each 16 cluster in y_cluster. use np.unique(y_cluster)
    ### generate 16 colors using the following code:
    my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_cell_type)}
    cell_type_color = [my_color[y_cell_type[i]] for i in range(len(y_cell_type))]

    return {'sample': sample_color, 'sex':sex_color, 'cell_type':cell_type_color}



def get_colors_dict_humanPBMC(y_sample, y_stim, y_cell_type):
     '''
     generate a dictionary of colors for each cell in the rat liver dataset
     y_sample: the sample for each cell
     y_sex: the sex info for each cell
     y_cell_type: the cluster for each cell
     '''
     ### make a dictionary of colors for each sample in y_sample
     my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_sample)}
     sample_color = [my_color[y_sample[i]] for i in range(len(y_sample))]

     ### make a dictionary of colors for each strain in y_strain
     my_color = {'stim': 'red', 'ctrl': 'blue'}
     stim_color = [my_color[y_stim[i]] for i in range(len(y_stim))]

     ### make a dictionary of colors for each 16 cluster in y_cluster. use np.unique(y_cluster)
     ### generate 16 colors using the following code:
     my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_cell_type)}
     cell_type_color = [my_color[y_cell_type[i]] for i in range(len(y_cell_type))]

     return {'sample': sample_color, 'stim':stim_color, 'cell_type':cell_type_color}

