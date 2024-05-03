from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from exutils import permutation as pmut
import numpy as np
import pandas as pd

from sciRED import rotations as rot
from sciRED.utils import preprocess as proc
from exutils import ex_preprocess as exproc
import time


np.random.seed(10)
NUM_COMPONENTS = 30
NUM_GENES = 2000


data_file_path = '/home/delaram/sciFA//Data/Human_Kidney_data.h5ad'  ### data is already normalized
data = exproc.import_AnnData(data_file_path)  ## based on scanpy

data, gene_idx = proc.get_sub_data(data, num_genes=NUM_GENES) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = proc.get_data_array(data)
y_sample, y_sex, y_cell_type, y_cell_type_sub = exproc.get_metadata_humanKidney(data)

num_levels_sample = len(y_sample.unique())
num_levels_cell_type = len(y_cell_type.unique())
num_levels_sex = len(y_sex.unique())

####################################
#### Running PCA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=NUM_COMPONENTS))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_ 
pca_loading.shape #(factors, genes)

####################################
#### Matching between factors and covariates ######
####################################
####### Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

factor_scores = pca_scores_varimax
scores_included = 'baseline'
residual_type = 'pearson'
n = 500

####################################
#### fcat score calculation and run time as baseline ######
fcat_dict_sample, time_dict_a_level_dict_sample = pmut.get_FCAT_dict(y_sample, factor_scores, time_eff=True) 
fcat_dict_cell_type, time_dict_a_level_dict_cell_type = pmut.get_FCAT_dict(y_cell_type, factor_scores, time_eff=True) 
fcat_dict_sex, time_dict_a_level_dict_sex = pmut.get_FCAT_dict(y_sex, factor_scores, time_eff=True)

covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type + ['sex']*num_levels_sex


time_df_dict = {**time_dict_a_level_dict_sample, **time_dict_a_level_dict_cell_type, **time_dict_a_level_dict_sex}
time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', 
                                columns=list(time_df_dict.values())[0].keys())
fcat_dict = {**fcat_dict_sample, **fcat_dict_cell_type, **fcat_dict_sex}
fcat_m = pmut.get_melted_fcat(fcat_dict)
fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]


meanimp_standard_arith_sample, meanimp_standard_geom_sample, meanimp_minmax_arith_sample, meanimp_minmax_geom_sample, meanimp_rank_arith_sample, meanimp_rank_geom_sample = pmut.get_mean_fcat_list(fcat_dict_sample)
meanimp_standard_arith_cell_type, meanimp_standard_geom_cell_type, meanimp_minmax_arith_cell_type, meanimp_minmax_geom_cell_type, meanimp_rank_arith_cell_type, meanimp_rank_geom_cell_type = pmut.get_mean_fcat_list(fcat_dict_cell_type)
meanimp_standard_arith_sex, meanimp_standard_geom_sex, meanimp_minmax_arith_sex, meanimp_minmax_geom_sex, meanimp_rank_arith_sex, meanimp_rank_geom_sex = pmut.get_mean_fcat_list(fcat_dict_sex)

meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_sample, meanimp_standard_arith_cell_type, meanimp_standard_arith_sex], axis=0)
meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_sample, meanimp_standard_geom_cell_type, meanimp_standard_geom_sex], axis=0)
meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_sample, meanimp_minmax_arith_cell_type, meanimp_minmax_arith_sex], axis=0)
meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_sample, meanimp_minmax_geom_cell_type, meanimp_minmax_geom_sex], axis=0)
meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_sample, meanimp_rank_arith_cell_type, meanimp_rank_arith_sex], axis=0)
meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_sample, meanimp_rank_geom_cell_type, meanimp_rank_geom_sex], axis=0)


meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                meanimp_rank_arith_df, meanimp_rank_geom_df]

mean_type_list = ['arithmatic', 'geometric', 
                'arithmatic', 'geometric', 
                'arithmatic', 'geometric']

scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']

scores_included_list = [scores_included]*len(meanimp_df_list)
covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type + ['sex']*num_levels_sex

meanimp_df = pmut.concatMeanFCAT(meanimp_df_list, mean_type_list, 
                                    scale_type_list, scores_included_list, 
                                    residual_type=residual_type, covariate_list=covariate_list)

### save baseline (unshuffled) fcat_m to csv
fcat_m.to_csv('/home/delaram/sciRED/benchmark/humanKidney/baseline/fcat_humanKidney_single_baseline.csv')
meanimp_df.to_csv('/home/delaram/sciRED/benchmark/humanKidney/baseline/fcat_humanKidney_mean_baseline.csv')


t_start_total = time.time()
#### shuffle the covariate vectors n times in a loop
for i in range(n):
    print('i: ', i)

    y_sample_shuffled = pmut.shuffle_covariate(y_sample)
    y_cell_type_shuffled = pmut.shuffle_covariate(y_cell_type)
    y_sex_shuffled = pmut.shuffle_covariate(y_sex)

    ####################################
    #### Importance calculation and run time for model comparison  ######
    ####################################
    fcat_dict_sample, time_dict_a_level_dict_sample =  pmut.get_FCAT_dict(y_sample_shuffled, factor_scores, time_eff=True) 
    fcat_dict_cell_type, time_dict_a_level_dict_cell_type =  pmut.get_FCAT_dict(y_cell_type_shuffled, factor_scores, time_eff=True) 
    fcat_dict_sex, time_dict_a_level_dict_sex = pmut.get_FCAT_dict(y_sex_shuffled, factor_scores, time_eff=True)

    ####################################
    ########### Comparing model run times
    ####################################
    time_df_dict = {**time_dict_a_level_dict_sample, **time_dict_a_level_dict_cell_type, **time_dict_a_level_dict_sex}
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
    time_df.to_csv('/home/delaram/sciRED/benchmark/humanKidney/time/fcat_humanKidney_time_'+str(i)+'.csv')
    
    ############################################################
    ########## Comparing factor scores between models
    ############################################################
    #### merge two fca_dict_sample and fca_dict_cell_type dicts
    fcat_dict = {**fcat_dict_sample, **fcat_dict_cell_type, **fcat_dict_sex}    

    fcat_m = pmut.get_melted_fcat(fcat_dict)
    fcat_m['residual_type'] = [residual_type]*fcat_m.shape[0]

    fcat_m.to_csv('/home/delaram/sciRED/benchmark/humanKidney/shuffle/single/fcat_humanKidney_single_shuffle_'+str(i)+'.csv')

    ############################################################
    ########## Comparing factor scores between ensembles
    ############################################################

    meanimp_standard_arith_sample, meanimp_standard_geom_sample, meanimp_minmax_arith_sample, meanimp_minmax_geom_sample, meanimp_rank_arith_sample, meanimp_rank_geom_sample = pmut.get_mean_fcat_list(fcat_dict_sample)
    meanimp_standard_arith_cell_type, meanimp_standard_geom_cell_type, meanimp_minmax_arith_cell_type, meanimp_minmax_geom_cell_type, meanimp_rank_arith_cell_type, meanimp_rank_geom_cell_type = pmut.get_mean_fcat_list(fcat_dict_cell_type)
    meanimp_standard_arith_sex, meanimp_standard_geom_sex, meanimp_minmax_arith_sex, meanimp_minmax_geom_sex, meanimp_rank_arith_sex, meanimp_rank_geom_sex = pmut.get_mean_fcat_list(fcat_dict_sex)

    meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_sample, meanimp_standard_arith_cell_type, meanimp_standard_arith_sex], axis=0)
    meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_sample, meanimp_standard_geom_cell_type, meanimp_standard_geom_sex], axis=0)
    meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_sample, meanimp_minmax_arith_cell_type, meanimp_minmax_arith_sex], axis=0)
    meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_sample, meanimp_minmax_geom_cell_type, meanimp_minmax_geom_sex], axis=0)
    meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_sample, meanimp_rank_arith_cell_type, meanimp_rank_arith_sex], axis=0)
    meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_sample, meanimp_rank_geom_cell_type, meanimp_rank_geom_sex], axis=0)

    meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                    meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                    meanimp_rank_arith_df, meanimp_rank_geom_df]

    mean_type_list = ['arithmatic', 'geometric', 
                    'arithmatic', 'geometric', 
                    'arithmatic', 'geometric']
    scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']
    scores_included_list = [scores_included]*len(meanimp_df_list)
    covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type + ['sex']*num_levels_sex
    meanimp_df = pmut.concatMeanFCAT(meanimp_df_list, mean_type_list, 
                                        scale_type_list, scores_included_list, 
                                        residual_type=residual_type, covariate_list=covariate_list)

    meanimp_df.to_csv('/home/delaram/sciRED/benchmark/humanKidney/shuffle/mean/fcat_humanKidney_mean_shuffle_'+str(i)+'.csv')



t_end_total = time.time()
print('Total time: ', t_end_total - t_start_total)


