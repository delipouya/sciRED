
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.pipeline import Pipeline

from sciRED import ensembleFCA as efca
from sciRED import glm
from sciRED import rotations as rot
from sciRED import metrics as met

from sciRED.utils import preprocess as proc
from sciRED.utils import visualize as vis
from sciRED.utils import corr
from sciRED.examples import ex_preprocess as exproc
from sciRED.examples import ex_visualize as exvis

NUM_COMPONENTS = 30
NUM_COMPONENTS = 128
factors_df = pd.read_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/spectra_factors_pbmc_numcomp_'+ str(NUM_COMPONENTS)+ '.csv')
markers_df = pd.read_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/spectra_markers_pbmc_numcomp_'+ str(NUM_COMPONENTS)+ '.csv')
cell_scores_df = pd.read_csv('/home/delaram/sciRED//review_analysis/benchmark_methods/spectra_cell_scores_pbmc_numcomp_'+ str(NUM_COMPONENTS)+ '.csv')

pattern_cols = [col for col in cell_scores_df.columns if 'F' in col]
spectra_scores_factor = cell_scores_df[pattern_cols]
factor_scores = cell_scores_df.to_numpy()

### remove factors that the sum scores of all cells is zero and 
## print the number of factors that are removed

### check the distribution of the sum of scores for each factor
factor_scores_sum = np.sum(factor_scores, axis=0)
plt.hist(factor_scores_sum, bins=30)
plt.xlabel('Sum of scores', fontsize=34)
plt.ylabel('Frequency', fontsize=34)
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.title('Distribution of the sum of scores for each factor')
plt.show()


sum_factor_threshold = 20
to_be_included = factor_scores_sum > sum_factor_threshold
factor_numbers = ['F'+str(i) for i in range(1, factor_scores.shape[1]+1)]
included_factor_numbers = [factor_numbers[i] for i in range(len(to_be_included)) if to_be_included[i]]
factor_scores = factor_scores[:, to_be_included]

print('Number of factors removed: ', cell_scores.shape[1] - factor_scores.shape[1])
print(factor_scores.shape)


### FCAT needs to be calculated for each covariate separately
fcat_sample = efca.FCAT(y_sample, factor_scores, scale='standard', mean='arithmatic')
fcat_cell_type = efca.FCAT(y_cell_type, factor_scores, scale='standard', mean='arithmatic')
fcat_stim = efca.FCAT(y_stim, factor_scores, scale='standard', mean='arithmatic')

### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_cell_type, fcat_stim, fcat_sample], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_label=included_factor_numbers,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)

### concatenate FCAT table for protocol and cell line
fcat = pd.concat([fcat_cell_type, fcat_stim], axis=0)
fcat = fcat[fcat.index != 'NA'] ### remove the rownames called NA from table
vis.plot_FCAT(fcat, title='', color='coolwarm',
              x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
              x_axis_label=included_factor_numbers,
              x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)




fcat = pd.concat([fcat_cell_type, fcat_stim, fcat_sample], axis=0)

### using Otsu's method to calculate the threshold
threshold = efca.get_otsu_threshold(fcat.values.flatten())

vis.plot_histogram(fcat.values.flatten(),
                   xlabel='Factor-Covariate Association scores',
                   title='FCAT score distribution',
                   threshold=threshold)


## rownames of the FCAT table
all_covariate_levels = fcat.index.values
matched_factor_dist, percent_matched_fact = efca.get_percent_matched_factors(fcat, threshold)
matched_covariate_dist, percent_matched_cov = efca.get_percent_matched_covariates(fcat, threshold=threshold)
print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
vis.plot_matched_factor_dist(matched_factor_dist)
vis.plot_matched_covariate_dist(matched_covariate_dist, 
                                covariate_levels=all_covariate_levels)


### select the factors that are matched with any covariate level
matched_factor_index = np.where(matched_factor_dist>0)[0] 
fcat_matched = fcat.iloc[:,matched_factor_index] 
x_labels_matched = fcat_matched.columns.values
included_factor_numbers_matched = [included_factor_numbers[i] for i in matched_factor_index]
vis.plot_FCAT(fcat_matched, title='', color='coolwarm',
                                 x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                                 x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, 
                                 x_axis_label=included_factor_numbers_matched,
                                 save=False, save_path='../Plots/mean_importance_df_matched_lung.pdf')



