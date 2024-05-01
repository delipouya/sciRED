library(Seurat)
library(SeuratData)
library(SeuratDisk)

meta_data = read.csv('~/sciFA/Data/kidney_meta.tsv', sep = '\t')
data = read.csv('~/sciFA/Data/kidney_exprMatrix.tsv', sep = '\t')
row.names(data)= data[,1]
data = data[,-1]

colnames =  gsub('\\.', '-', colnames(data))
colnames(data) = as.character(colnames)

data_seur = CreateSeuratObject(data)
length(colnames(data_seur))
length(meta_data$Cell)
sum(colnames(data_seur) != meta_data$Cell)

data_seur@meta.data = cbind(data_seur@meta.data, meta_data)
head(data_seur@meta.data )
saveRDS(data_seur, '~/scLMM/LMM-scRNAseq/Data/Human_Kidney_data.rds')

SaveH5Seurat(data_seur, filename = "~/sciFA/Data/Human_Kidney_data.h5Seurat")
Convert("~/sciFA/Data/Human_Kidney_data.h5Seurat", dest = "h5ad")

