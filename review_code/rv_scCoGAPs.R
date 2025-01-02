#### Running NIFA on the immune population of the new samples ####
# Load required libraries
#source('/home/delaram/RatLiver/Codes/Functions.R')
library(BiocParallel)
library(devtools)
library(CoGAPS)
#library(NIFA)
library(fastICA)
library(Seurat)
library(parallel)  # Load the parallel package for detectCores()


# Load the scRNA-seq data for the merged samples
scMix_3cl_merged <- readRDS('/home/delaram/scLMM/LMM-scRNAseq/Data/scMix_3cl_merged_logcounts.rds')
merged_samples <- scMix_3cl_merged

#################### scCoGAPS Analysis ####################
# Set up CoGAPS parameters
params <- new("CogapsParams")
num_patterns <- 30

start_time <- proc.time()

params <- setParam(params, "nPatterns", num_patterns) # Set number of patterns
params <- setDistributedParams(params, nSets = detectCores() - 2) # Use available cores
getParam(params, "nPatterns")

########### Prepare input data ########### 
# Convert the Seurat object to matrix format
data <- as.matrix(GetAssayData(merged_samples))
Results.sc <- CoGAPS(data, params, distributed = "single-cell", messages = TRUE, transposeData = FALSE)

end_time <- proc.time()
runtime <- end_time - start_time
print(paste0( 'run time for scCoGAPs - scMix - #F: ', num_patterns, ' is: ',runtime[3]))


# Save the CoGAPS results
#saveRDS(Results.sc, '/home/delaram/sciRED/review_analysis/scCoGAPS_result_scMix.rds')
Results.sc = readRDS('/home/delaram/sciRED/review_analysis/scCoGAPS_result_scMix.rds')
head(Results.sc@sampleFactors)
head(Results.sc@featureLoadings)
head(merged_samples@meta.data)
sum(rownames(merged_samples@meta.data) != rownames(Results.sc@sampleFactors))
scCoGAPS_scores = data.frame(Results.sc@sampleFactors)
dim(scCoGAPS_scores)

scCoGAPS_scores = cbind(scCoGAPS_scores, merged_samples@meta.data)
scCoGAPS_loading = data.frame(Results.sc@featureLoadings)
head(Results.sc@featureLoadings)
write.csv(scCoGAPS_loading, '/home/delaram/sciRED//review_analysis/scCoGAPS_loading_scMix.csv')
write.csv(scCoGAPS_scores, '/home/delaram/sciRED//review_analysis//scCoGAPS_scores_scMix.csv')


# Clean up memory
rm(data)
rm(Results.sc)
gc()
################### Load PBMC Data ###################
# Load normalized PBMC Lupus data (pre-processed using Seurat)
Kang18_8vs8_seur <- readRDS('/home/delaram/scLMM/LMM-scRNAseq/Data/PBMC_Lupus_Kang8vs8_data_norm.rds')
merged_samples <- Kang18_8vs8_seur

# Set up CoGAPS parameters for PBMC data
params <- new("CogapsParams")
num_patterns <- 30
params <- setParam(params, "nPatterns", num_patterns)
params <- setDistributedParams(params, nSets = detectCores() - 2)
getParam(params, "nPatterns")


########### Prepare input data ########### 
# Convert the Seurat object to matrix format
data <- as.matrix(GetAssayData(merged_samples))

# Run CoGAPS on the PBMC data
Results.sc <- CoGAPS(data, params, distributed = "single-cell", messages = TRUE, transposeData = FALSE)

# Save the CoGAPS results for PBMC
#saveRDS(Results.sc, '/home/delaram/sciRED/review_analysis/scCoGAPS_result_pbmc.rds')
Results.sc = readRDS('/home/delaram/sciRED/review_analysis/scCoGAPS_result_pbmc.rds')
## worker 11 is finished! Time: 39:56:05
#### Run time is 39 hours!!!
head(Results.sc@sampleFactors)



Results.sc = readRDS('/home/delaram/sciRED/review_analysis/scCoGAPS_result_pbmc.rds')
head(Results.sc@sampleFactors)
head(Results.sc@featureLoadings)
head(merged_samples@meta.data)
sum(rownames(merged_samples@meta.data) != rownames(Results.sc@sampleFactors))
scCoGAPS_scores = data.frame(Results.sc@sampleFactors)
dim(scCoGAPS_scores)

scCoGAPS_scores = cbind(scCoGAPS_scores, merged_samples@meta.data)
scCoGAPS_loading = data.frame(Results.sc@featureLoadings)
head(Results.sc@featureLoadings)
write.csv(scCoGAPS_loading, '/home/delaram/sciRED//review_analysis/scCoGAPS_loading_PBMC.csv')
write.csv(scCoGAPS_scores, '/home/delaram/sciRED//review_analysis//scCoGAPS_scores_PBMC.csv')

