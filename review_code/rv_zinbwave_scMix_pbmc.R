#BiocManager::install("zinbwave")
#remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)
library(zinbwave)
library(matrixStats)
library(magrittr)
library(ggplot2)
library(biomaRt)
library(sparseMatrixStats)
# install scater https://bioconductor.org/packages/release/bioc/html/scater.html
library(Seurat)

# Register BiocParallel Serial Execution
BiocParallel::register(BiocParallel::SerialParam())

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
merged_samples = merged_samples[rowSums(merged_samples)>1,]
merged_samples.sc <- as.SingleCellExperiment(merged_samples)

num_dims = 10
num_genes = 2000

scMix_zinb <- zinbwave(merged_samples.sc, K = num_dims, epsilon=num_genes)
W <- reducedDim(scMix_zinb)
saveRDS(list('scMix_zinb'=scMix_zinb, 'W'=W), 
        paste0('~/sciRED/review_analysis/benchmark_methods/zinbwave_result_scMix_numcomp_',num_dims,'.rds'))



res = readRDS(paste0('~/sciRED/review_analysis/benchmark_methods/zinbwave_result_scMix_numcomp_',num_dims,'.rds'))
merged_samples <- readRDS('/home/delaram/scLMM/LMM-scRNAseq/Data/scMix_3cl_merged_logcounts.rds')
table(merged_samples$sample)

scMix_zinb = res$scMix_zinb
scMix_metadata = data.frame(merged_samples@meta.data) #res$scMix_zinb@colData
sum(colnames(merged_samples) != rownames(scMix_metadata))

zinb_scores = data.frame(res$W)
sum(rownames(zinb_scores) != rownames(scMix_metadata))
zinb_scores = cbind(zinb_scores, scMix_metadata)
head(zinb_scores)

#write.csv(zinb_scores, paste0('/home/delaram/sciRED//review_analysis/benchmark_methods/zinbwave_scores_scMix_numcomp_',num_dims,'.csv'))





###################### PBMC dataset ######################
Kang18_8vs8_seur = readRDS("~/scLMM/LMM-scRNAseq//Data/PBMC_Lupus_Kang8vs8_data_counts.rds")
merged_samples <- Kang18_8vs8_seur
merged_samples = merged_samples[rowSums(merged_samples)>1,]
merged_samples.sc <- as.SingleCellExperiment(merged_samples)

num_dims = 30
num_genes = 2000
pbmc_zinb <- zinbwave(merged_samples.sc, K = num_dims, epsilon=num_genes)
W <- reducedDim(pbmc_zinb)
saveRDS(list('pbmc_zinb'=pbmc_zinb, 'W'=W), 
        '~/sciRED/review_analysis/zinbwave_result_pbmc.rds')




