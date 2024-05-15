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
import time

np.random.seed(10)
NUM_COMPONENTS = 30
NUM_GENES = 2000
NUM_COMP_TO_VIS = 5


data_file_path = '/home/delaram/sciFA/Data/scMix_3cl_merged.h5ad'
data = exproc.import_AnnData(data_file_path)
data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)
y_cell_line, y_sample, y_protocol = exproc.get_metadata_scMix(data)
data.obs['protocol'] = y_protocol.to_numpy()
data.obs['cell_line'] = y_cell_line.to_numpy()
data.obs['sample'] = y_sample.to_numpy()


####################################
#### fit GLM to each gene ######
####################################
#### design matrix - library size only
x = proc.get_library_design_mat(data, lib_size='nCount_originalexp')
glm_fit_dict = glm.poissonGLM(y, x)
residual_type = 'pearson'
resid = glm_fit_dict['resid_pearson'] 
y = resid.T 

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
fcat_dict_protocol, time_dict_a_level_dict_protocol = pmut.get_FCAT_dict(y_protocol, factor_scores, time_eff=True) 
fcat_dict_cell_line, time_dict_a_level_dict_cell_line = pmut.get_FCAT_dict(y_cell_line, factor_scores, time_eff=True) 
covariate_list = ['protocol']*3 + ['cell_line']*3


time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', 
                                columns=list(time_df_dict.values())[0].keys())
fcat_dict = {**fcat_dict_protocol, **fcat_dict_cell_line}
fcat_m = pmut.get_melted_fcat(fcat_dict)
fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]


meanimp_standard_arith_protocol, meanimp_standard_geom_protocol, meanimp_minmax_arith_protocol, meanimp_minmax_geom_protocol, meanimp_rank_arith_protocol, meanimp_rank_geom_protocol = pmut.get_mean_fcat_list(fcat_dict_protocol)
meanimp_standard_arith_cell_line, meanimp_standard_geom_cell_line, meanimp_minmax_arith_cell_line, meanimp_minmax_geom_cell_line, meanimp_rank_arith_cell_line, meanimp_rank_geom_cell_line = pmut.get_mean_fcat_list(fcat_dict_cell_line)

meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_protocol, meanimp_standard_arith_cell_line], axis=0)
meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_protocol, meanimp_standard_geom_cell_line], axis=0)
meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_protocol, meanimp_minmax_arith_cell_line], axis=0)
meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_protocol, meanimp_minmax_geom_cell_line], axis=0)
meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_protocol, meanimp_rank_arith_cell_line], axis=0)
meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_protocol, meanimp_rank_geom_cell_line], axis=0)


meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                meanimp_rank_arith_df, meanimp_rank_geom_df]

mean_type_list = ['arithmatic', 'geometric', 
                'arithmatic', 'geometric', 
                'arithmatic', 'geometric']

scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']

scores_included_list = [scores_included]*len(meanimp_df_list)
covariate_list = ['protocol']*3 + ['cell_line']*3

meanimp_df = pmut.concatMeanFCAT(meanimp_df_list, mean_type_list, 
                                    scale_type_list, scores_included_list, 
                                    residual_type=residual_type, covariate_list=covariate_list)

### save baseline (unshuffled) fcat_m to csv
fcat_m.to_csv('/home/delaram/sciRED/benchmark/scMix/baseline/fcat_scMix_single_baseline.csv')
meanimp_df.to_csv('/home/delaram/sciRED/benchmark/scMix/baseline/fcat_scMix_mean_baseline.csv')


t_start_total = time.time()
#### shuffle the covariate vectors n times in a loop
for i in range(n):
    print('i: ', i)

    y_protocol_shuffled = pmut.shuffle_covariate(y_protocol)
    y_cell_line_shuffled = pmut.shuffle_covariate(y_cell_line)

    ####################################
    #### Importance calculation and run time for model comparison  ######
    ####################################
    fcat_dict_protocol, time_dict_a_level_dict_protocol =  pmut.get_FCAT_dict(y_protocol_shuffled, factor_scores, time_eff=True) 
    fcat_dict_cell_line, time_dict_a_level_dict_cell_line =  pmut.get_FCAT_dict(y_cell_line_shuffled, factor_scores, time_eff=True) 
    
    ####################################
    ########### Comparing model run times
    ####################################
    time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
    time_df.to_csv('/home/delaram/sciRED/benchmark/scMix/time/fcat_scMix_time_'+str(i)+'.csv')
    
    ############################################################
    ########## Comparing factor scores between models
    ############################################################
    #### merge two fca_dict_protocol and fca_dict_cell_line dicts
    fca_dict = {**fcat_dict_protocol, **fcat_dict_cell_line}
    fcat_m = pmut.get_melted_fcat(fca_dict)
    fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]

    fcat_m.to_csv('/home/delaram/sciRED/benchmark/scMix/shuffle/single/fcat_scMix_single_shuffle_'+str(i)+'.csv')

    ############################################################
    ########## Comparing factor scores between ensembles
    ############################################################
    meanimp_standard_arith_protocol, meanimp_standard_geom_protocol, meanimp_minmax_arith_protocol, meanimp_minmax_geom_protocol, meanimp_rank_arith_protocol, meanimp_rank_geom_protocol = pmut.get_mean_fcat_list(fcat_dict_protocol)
    meanimp_standard_arith_cell_line, meanimp_standard_geom_cell_line, meanimp_minmax_arith_cell_line, meanimp_minmax_geom_cell_line, meanimp_rank_arith_cell_line, meanimp_rank_geom_cell_line = pmut.get_mean_fcat_list(fcat_dict_cell_line)

    meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_protocol, meanimp_standard_arith_cell_line], axis=0)
    meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_protocol, meanimp_standard_geom_cell_line], axis=0)
    meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_protocol, meanimp_minmax_arith_cell_line], axis=0)
    meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_protocol, meanimp_minmax_geom_cell_line], axis=0)
    meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_protocol, meanimp_rank_arith_cell_line], axis=0)
    meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_protocol, meanimp_rank_geom_cell_line], axis=0)

    meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                    meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                    meanimp_rank_arith_df, meanimp_rank_geom_df]

    mean_type_list = ['arithmatic', 'geometric', 
                    'arithmatic', 'geometric', 
                    'arithmatic', 'geometric']
    scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']
    scores_included_list = [scores_included]*len(meanimp_df_list)
    covariate_list = ['protocol']*3 + ['cell_line']*3
    meanimp_df = pmut.concatMeanFCAT(meanimp_df_list, mean_type_list, 
                                        scale_type_list, scores_included_list, 
                                        residual_type=residual_type, covariate_list=covariate_list)

    meanimp_df.to_csv('/home/delaram/sciRED/benchmark/scMix/shuffle/mean/fcat_scMix_mean_shuffle_'+str(i)+'.csv')



t_end_total = time.time()
print('Total time: ', t_end_total - t_start_total)


