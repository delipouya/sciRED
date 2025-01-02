#### Running NIFA on the immune population of the new samples ####
# Load required libraries
#source('/home/delaram/RatLiver/Codes/Functions.R')
library(BiocParallel)
library(devtools)
library(CoGAPS)
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




################### Load PBMC Data ###################
# Load normalized PBMC Lupus data (pre-processed using Seurat)
Kang18_8vs8_seur <- readRDS('/home/delaram/scLMM/LMM-scRNAseq/Data/PBMC_Lupus_Kang8vs8_data_norm.rds')
merged_samples <- Kang18_8vs8_seur
params <- new("CogapsParams")
num_patterns <- 30

start_time <- proc.time()

params <- setParam(params, "nPatterns", num_patterns)
params <- setDistributedParams(params, nSets = detectCores() - 2)
getParam(params, "nPatterns")
data <- as.matrix(GetAssayData(merged_samples))
Results.sc <- CoGAPS(data, params, distributed = "single-cell", messages = TRUE, transposeData = FALSE)

end_time <- proc.time()
runtime <- end_time - start_time
print(paste0( 'run time for scCoGAPs - pbmc - #F: ', num_patterns, ' is: ',runtime[3]))



