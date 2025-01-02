#### ZINB-WaVE Analysis on PBMC Dataset ####
# Load required libraries
library(zinbwave)
library(matrixStats)
library(magrittr)
library(ggplot2)
library(biomaRt)
library(sparseMatrixStats)
library(Seurat)
library(scater)  # scater is assumed to be installed as per your comment

# Register BiocParallel Serial Execution
BiocParallel::register(BiocParallel::SerialParam())

###################### PBMC Dataset ######################
# Load PBMC Lupus data
#Kang18_8vs8_seur <- readRDS('/home/delaram/scLMM/LMM-scRNAseq/Data/PBMC_Lupus_Kang8vs8_data_counts.rds')
#merged_samples <- Kang18_8vs8_seur


###################### scMixology dataset ######################
load("~/scLMM/sc_mixology/data//sincell_with_class.RData") ## 3 cell line data
data_list_3cl = list(sc_10x=sce_sc_10x_qc, 
                     CELseq2=sce_sc_CELseq2_qc, 
                     Dropseq=sce_sc_Dropseq_qc)

data_list_3cl = lapply(data_list_3cl, function(x) as.Seurat(x, counts = "counts", data = "counts")) 
names(data_list_3cl) = c('sc_10X', 'CELseq2', 'Dropseq')
scMix_3cl_merged <- merge(data_list_3cl[[1]], c(data_list_3cl[[2]], data_list_3cl[[3]]),
                          add.cell.ids = names(data_list_3cl), 
                          project = "scMix_3cl", 
                          merge.data = TRUE)

merged_samples <- scMix_3cl_merged

# Filter genes with row sums > 1
merged_samples <- merged_samples[rowSums(merged_samples) > 1, ]

# Convert Seurat object to SingleCellExperiment
merged_samples.sc <- as.SingleCellExperiment(merged_samples)

# Set parameters for ZINB-WaVE
num_dims <- 30
num_genes <- 2000

start_time <- proc.time()
# Run ZINB-WaVE
pbmc_zinb <- zinbwave(merged_samples.sc, K = num_dims, epsilon = num_genes)

# Extract reduced dimensions (W)
W <- reducedDim(pbmc_zinb)

end_time <- proc.time()
runtime <- end_time - start_time
print(paste0( 'run time for zinbwave - scMix - #F: ', num_dims, ' is: ',runtime[3]))

# Save ZINB-WaVE result
#saveRDS(list('pbmc_zinb' = pbmc_zinb, 'W' = W), 
#        '/home/delaram/sciRED/review_analysis/zinbwave_result_pbmc.rds')


# run as Rscript ./rv_zinbwave.R > rv_zinbwave.log



