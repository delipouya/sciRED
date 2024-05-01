######## preparing the data within the Muscat dataset 
# 10x droplet-based scRNA-seq PBMC data from 8 Lupus patients before and after 6h-treatment with INF-beta 
# https://github.com/HelenaLC/muscData
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583

#BiocManager::install("muscData")
library(muscData)
library(Seurat)
library(SeuratData)
library(SeuratDisk)

Kang18_8vs8 = muscData::Kang18_8vs8(metadata = FALSE)
Kang18_8vs8$multiplets
Kang18_8vs8@assays[['counts']]
class(Kang18_8vs8)
Kang18_8vs8_seur = as.Seurat(Kang18_8vs8, counts = "counts", data = "counts")
GetAssayData(Kang18_8vs8_seur)[100:120,100:150]

Kang18_8vs8_seur$stim = as.character(Kang18_8vs8_seur$stim)
Kang18_8vs8_seur$ind = as.character(Kang18_8vs8_seur$ind)
Kang18_8vs8_seur$cluster = as.character(Kang18_8vs8_seur$cluster)
Kang18_8vs8_seur$cell = as.character(Kang18_8vs8_seur$cell)
Kang18_8vs8_seur$multiplets = as.character(Kang18_8vs8_seur$multiplets)
saveRDS(Kang18_8vs8_seur, file = "~/scLMM/LMM-scRNAseq//Data/PBMC_Lupus_Kang8vs8_data_counts.rds")

Kang18_8vs8_seur <- Seurat::SCTransform(Kang18_8vs8_seur, return.only.var.genes = FALSE, 
                                        conserve.memory = FALSE, 
                                        variable.features.n = nrow(Kang18_8vs8_seur),assay = 'originalexp')

saveRDS(Kang18_8vs8_seur, file = "~/scLMM/LMM-scRNAseq//Data/PBMC_Lupus_Kang8vs8_data_norm.rds")

SaveH5Seurat(Kang18_8vs8_seur, filename = "~/sciFA/Data/PBMC_Lupus_Kang8vs8_data.h5Seurat")
Convert("~/sciFA/Data/PBMC_Lupus_Kang8vs8_data.h5Seurat", dest = "h5ad")


