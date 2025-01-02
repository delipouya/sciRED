setwd('~/scLMM/sc_mixology/')
library(scran)
library(scater)
library(ggplot2)
library(DT)
library(Rtsne)
#remotes::install_github(repo ='mojaveazure/loomR', ref = 'develop')
library(loomR)
library(Seurat)
library(patchwork)
library(SeuratData)
library(SeuratDisk)

set.seed(5252)

###########################################################################
############### Merging the 3cl data to be imported to python #############
###########################################################################

load("~/scLMM/sc_mixology/data//sincell_with_class.RData") ## 3 cell line data
#sce10x_qc contains the read counts after quality control processing from the 10x platform. 
#sce4_qc contains the read counts after quality control processing from the CEL-seq2 platform. 
#scedrop_qc_qc contains the read counts after quality control proessing from the Drop-seq platform.
data_list_3cl = list(sc_10x=sce_sc_10x_qc, 
                     CELseq2=sce_sc_CELseq2_qc, 
                     Dropseq=sce_sc_Dropseq_qc)

data_list_3cl = lapply(data_list_3cl, function(x) as.Seurat(x, counts = "counts", data = "counts")) 
#data_list_3cl = lapply(data_list_3cl, function(x) as.Seurat(x, counts = "logcounts", data = "logcounts")) 

data_list_3cl = lapply(data_list_3cl, function(x) {SCTransform(x, variable.features.n = nrow(x),assay='originalexp')}) 
names(data_list_3cl) = c('sc_10X', 'CELseq2', 'Dropseq')
data_list_3cl = sapply(1:length(data_list_3cl), 
                       function(i) {data_list_3cl[[i]]$sample=names(data_list_3cl)[i]; data_list_3cl[[i]]}, simplify = F)
names(data_list_3cl) = c('sc_10X', 'CELseq2', 'Dropseq')

scMix_3cl_merged <- merge(data_list_3cl[[1]], c(data_list_3cl[[2]], data_list_3cl[[3]]),
                          add.cell.ids = names(data_list_3cl), 
                          project = "scMix_3cl", 
                          merge.data = TRUE)

saveRDS(scMix_3cl_merged, file = '~/scLMM/LMM-scRNAseq/Data/scMix_3cl_merged_sctransform.rds')
saveRDS(scMix_3cl_merged, file = '~/scLMM/LMM-scRNAseq/Data/scMix_3cl_merged_logcounts.rds')
SaveH5Seurat(scMix_3cl_merged, filename = "~/scLMM/sc_mixology/scMix_3cl_merged.h5Seurat")
Convert("~/scLMM/sc_mixology/scMix_3cl_merged.h5Seurat", dest = "h5ad")

###########################################################################
############### Merging the 5cl data to be imported to python #############
###########################################################################
rm(list=ls())
load("~/sciFA/data/sincell_with_class_5cl.RData") ## 5 cell line data
data_list_5cl = list(sc_10x=sce_sc_10x_5cl_qc, 
                     CELseq2_p1=sc_Celseq2_5cl_p1, 
                     CELseq2_p2=sc_Celseq2_5cl_p2,
                     CELseq2_p3=sc_Celseq2_5cl_p3)

data_list_5cl = lapply(data_list_5cl, function(x) logNormCounts(x))
data_list_5cl = lapply(data_list_5cl, function(x) as.Seurat(x, data = "counts")) # counts = "counts",
data_list_5cl_names = names(data_list_5cl)
data_list_5cl = sapply(1:length(data_list_5cl), 
                       function(i) {data_list_5cl[[i]]$sample=names(data_list_5cl)[i]; data_list_5cl[[i]]}, simplify = F)
names(data_list_5cl) = data_list_5cl_names

scMix_5cl_merged <- merge(data_list_5cl[[1]], c(data_list_5cl[[2]], data_list_5cl[[3]], data_list_5cl[[4]]),
                          add.cell.ids = names(data_list_5cl), 
                          project = "scMix_5cl", 
                          merge.data = TRUE)

SaveH5Seurat(scMix_5cl_merged, filename = "~/scLMM/sc_mixology/scMix_5cl_merged.h5Seurat")
Convert("~/scLMM/sc_mixology/scMix_5cl_merged.h5Seurat", dest = "h5ad")



