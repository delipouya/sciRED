import os
import pandas as pd
import numpy as np
from scipy.io import mmread
import scipy.sparse as sp
import matplotlib.pyplot as plt
from IPython.display import Image
import scanpy as sc
from cnmf import cNMF
import time

np.random.seed(10)

#### import the data
data_file_path = '/home/delaram/sciRED/Data/scMix_3cl_merged.h5ad'
data = sc.read(data_file_path) ## attributes removed
data.var_names_make_unique()
data.obs.head()
### renaming the meta info column names: https://github.com/theislab/scvelo/issues/255
data.__dict__['_raw'].__dict__['_var'] = data.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

## Save the scanpy object to a file. This file will be passed as input to cNMF
count_adat_fn = '/home/delaram/sciRED/review_analysis/benchmark_methods/cNMF/counts_scMix.h5ad'
#sc.write(count_adat_fn, data)


numiter=100 
numworkers = 1
#numworkers = os.cpu_count()-1
count_adat_fn = '/home/delaram/sciRED/review_analysis/benchmark_methods/cNMF/counts_scMix.h5ad'
## Results will be saved to [output_directory]/[run_name]
output_directory = '/home/delaram/sciRED/review_analysis/benchmark_methods/cNMF/'
run_name = 'scMix_cNMF_v2'
seed = 10 
K = np.array([10, 30, 50])
NUM_COMPONENTS = 30


### evaluate run time for fitting the model
start = time.time()
cnmf_obj = cNMF(output_dir=output_directory, name=run_name)
cnmf_obj.prepare(counts_fn=count_adat_fn, components=K, n_iter=numiter, seed=seed)
cnmf_obj.factorize(worker_i=0, total_workers=numworkers)
cnmf_obj.combine()
cnmf_obj.k_selection_plot()
cnmf_obj.consensus(k=NUM_COMPONENTS, density_threshold=0.01)
usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=NUM_COMPONENTS, 
                                                                      density_threshold=0.01)
end = time.time()

print('Time taken: ', end - start)

### replace the column names as F1, F2, ...
usage.columns = ['F'+str(i+1) for i in range(usage.shape[1])]
spectra_scores.columns = ['F'+str(i+1) for i in range(spectra_scores.shape[1])]
spectra_tpm.columns = ['F'+str(i+1) for i in range(spectra_tpm.shape[1])]
top_genes.columns = ['F'+str(i+1) for i in range(top_genes.shape[1])]

### save the results
usage.to_csv(os.path.join(output_directory, run_name + '_numcomp_'+str(NUM_COMPONENTS)+'_usage.csv'))
spectra_scores.to_csv(os.path.join(output_directory, run_name + '_numcomp_'+str(NUM_COMPONENTS)+'_spectra_scores.csv'))
spectra_tpm.to_csv(os.path.join(output_directory, run_name + '_numcomp_'+str(NUM_COMPONENTS)+'_spectra_tpm.csv'))
top_genes.to_csv(os.path.join(output_directory, run_name + '_numcomp_'+str(NUM_COMPONENTS)+'_top_genes.csv'))

print('Time taken (second): ', end - start)
print('Time taken (min): ', (end - start)/60)

#Time taken:  6059.3251230716705
#Time taken (second):  6059.3251230716705
#Time taken (min):  100.98875205119451