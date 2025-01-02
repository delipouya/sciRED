library(SingleCellExperiment)
library("zellkonverter")
library(SeuratData)
library(SeuratDisk)
library(Matrix)
library(zellkonverter)
library(Seurat)
library(zellkonverter)
library(SeuratDisk)


setwd('~/sciRED/review_analysis/')
download.file("https://cf.10xgenomics.com/samples/cell-exp/4.0.0/Parent_NGSC3_DI_PBMC/Parent_NGSC3_DI_PBMC_filtered_feature_bc_matrix.h5",
              destfile = "3p_pbmc10k_filt.h5")
download.file("https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5",
              destfile = "5p_pbmc10k_filt.h5")

# Path to your .h5ad file
h5ad_file <- '/home/delaram/sciRED/review_analysis/3p_pbmc10k_filt.h5ad'
sce_3pr <- readH5AD(h5ad_file)
duplicated_genes <- rownames(sce_3pr)[duplicated(rownames(sce_3pr))]
rownames(sce_3pr) <- make.unique(rownames(sce_3pr))
any(duplicated(rownames(sce_3pr)))
seur_3pr <- CreateSeuratObject(counts = assay(sce_3pr, "X"))
print(seur_3pr)

h5ad_file <- '/home/delaram/sciRED/review_analysis/5p_pbmc10k_filt.h5ad'
sce_5pr <- readH5AD(h5ad_file)
duplicated_genes <- rownames(sce_5pr)[duplicated(rownames(sce_5pr))]
rownames(sce_5pr) <- make.unique(rownames(sce_5pr))
any(duplicated(rownames(sce_5pr)))
seur_5pr <- CreateSeuratObject(counts = assay(sce_5pr, "X"))
print(seur_5pr)

seur_3pr$data='3p_pbmc10k'
Idents(seur_3pr) = '3p_pbmc10k'

seur_5pr$data='5p_pbmc10k'
Idents(seur_5pr) = '5p_pbmc10k'

# Merge the two Seurat objects
merged_seur <- merge(seur_3pr, y = seur_5pr, add.cell.ids = c("3p_pbmc10k", "5p_pbmc10k"), project = "PBMC10K")
Layers(merged_seur[["RNA"]])
merged_seur[["RNA"]] <- JoinLayers(merged_seur[["RNA"]])
Layers(merged_seur[["RNA"]])

print(merged_seur) 
dim(merged_seur) # 36601 20742
dim(seur_3pr) # 36601 10194
dim(seur_5pr) # 36601 10548
table(merged_seur$data)

sparse_mat = GetAssayData(merged_seur)
metadata <- merged_seur[[]]


write.csv(metadata, "~/sciRED/review_analysis/PBMC10K_3p5p_metadata.csv")
metadata1 = read.csv( "~/sciRED/review_analysis/PBMC10K_3p5p_metadata.csv")
write.csv(rownames(merged_seur), "~/sciRED/review_analysis/PBMC10K_3p5p_row_names.csv", row.names = FALSE)
write.csv(colnames(merged_seur), "~/sciRED/review_analysis/PBMC10K_3p5p_col_names.csv", row.names = FALSE)
######################################################

######################################################
################# writing the data into a file using npz and scipy directly
library(reticulate)
# Convert sparse matrix to Python object and save as .npz
scipy <- import("scipy.sparse")
np <- import("numpy")
scipy$save_npz(path.expand("~/sciRED/review_analysis/PBMC10K_3p5p_matrix.npz"), sparse_mat)


######################################################
merged_seur[["RNA"]] <- split(merged_seur[["RNA"]], f = merged_seur$data)
merged_seur
# run standard anlaysis workflow
merged_seur <- NormalizeData(merged_seur)
merged_seur <- FindVariableFeatures(merged_seur)
merged_seur <- ScaleData(merged_seur)
merged_seur <- RunPCA(merged_seur)
merged_seur <- FindNeighbors(merged_seur, dims = 1:30, reduction = "pca")
merged_seur <- FindClusters(merged_seur, resolution = 1, cluster.name = "unintegrated_clusters")
merged_seur <- RunUMAP(merged_seur, dims = 1:30, reduction = "pca", reduction.name = "umap.unintegrated")
DimPlot(merged_seur, reduction = "umap.unintegrated", group.by = c("data", "seurat_clusters"))


###### integration
#merged_seur <- IntegrateLayers(
#  object = merged_seur, method = CCAIntegration, 
#  orig.reduction = "pca", new.reduction = "integrated.cca",
#  verbose = FALSE)

merged_seur <- IntegrateLayers(
  object = merged_seur, method = HarmonyIntegration,
  orig.reduction = "pca", new.reduction = "harmony",
  verbose = FALSE
)



# re-join layers after integration
merged_seur[["RNA"]] <- JoinLayers(merged_seur[["RNA"]])

merged_seur <- FindNeighbors(merged_seur, reduction = "harmony", dims = 1:30)
merged_seur <- FindClusters(merged_seur, resolution = 0.5)
merged_seur <- RunUMAP(merged_seur, dims = 1:30, reduction = "harmony")
merged_seur$seurat_clusters
DimPlot(merged_seur, reduction = "umap", group.by = c("data", "seurat_clusters"))
DimPlot(merged_seur, reduction = "umap", split.by = "data")




# Load your PBMC dataset (already integrated)
# Assuming merged_seur is your integrated dataset
# Loading a pre-annotated PBMC reference dataset-  an example PBMC atlas from SeuratData 
InstallData("pbmc3k") 
pbmc_ref <- LoadData("pbmc3k", type = "pbmc3k.final") # Load the PBMC reference
pbmc_ref <- NormalizeData(pbmc_ref)
pbmc_ref <- FindVariableFeatures(pbmc_ref)
pbmc_ref <- ScaleData(pbmc_ref)
pbmc_ref <- RunPCA(pbmc_ref, features = VariableFeatures(object = pbmc_ref))

# Find anchors between the query and reference datasets
anchors <- FindTransferAnchors(
  reference = pbmc_ref, 
  query = merged_seur, 
  dims = 1:30
)

# Transfer the annotations from the reference to your dataset
merged_seur = TransferData(query = merged_seur,
  anchorset = anchors, 
  refdata = pbmc_ref$seurat_annotations, # Use the annotations in the reference dataset
  dims = 1:30
)

merged_seur <- RunUMAP(merged_seur, reduction = "pca", dims = 1:30)
merged_seur$predicted.id
DimPlot(merged_seur, reduction = "umap", group.by = "predicted.id", label = TRUE, repel = TRUE)
DimPlot(merged_seur, reduction = "umap", group.by = "predicted.id", label = TRUE, repel = TRUE, 
        cols = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'))
# Visualize split by dataset origin (e.g., 3' and 5' data)
DimPlot(merged_seur, reduction = "umap", split.by = "data")


dim(metadata1)
dim(merged_seur@meta.data)
sum(metadata1$X != rownames(merged_seur@meta.data))


metadata2 <- merged_seur@meta.data
metadata2$barcode <- rownames(metadata2)
pca_embeddings <- as.data.frame(Embeddings(merged_seur, reduction = "pca"))
umap_unintegrated_embeddings <- as.data.frame(Embeddings(merged_seur, reduction = "umap.unintegrated"))
harmony_embeddings <- as.data.frame(Embeddings(merged_seur, reduction = "harmony"))
umap_embeddings <- as.data.frame(Embeddings(merged_seur, reduction = "umap"))

metadata2 <- cbind(metadata2, pca_embeddings)
metadata2 <- cbind(metadata2, umap_unintegrated_embeddings)
metadata2 <- cbind(metadata2, harmony_embeddings)
metadata2 <- cbind(metadata2, umap_embeddings)

write.csv(metadata2, "~/sciRED/review_analysis/PBMC10K_3p5p_metadata_complete.csv")
metadata2 <- read.csv("~/sciRED/review_analysis/PBMC10K_3p5p_metadata_complete.csv")

library(ggplot2)
library(RColorBrewer)
color_palette <- brewer.pal(n = 12, name = "Set3")

my_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22')
ggplot(metadata2, aes(x = umapunintegrated_1, y = umapunintegrated_2, color = predicted.id)) +
  geom_point(size = 1, alpha = 0.7) +  # Adjust point size and transparency
  #scale_color_brewer(palette = "Set1") +  # Use the Set3 color palette
  scale_color_manual(values = my_colors) +
  theme_classic() +  # Use a clean thmy_coloeme
  theme(
    axis.text.x = element_text(size = 12),  # Make x-axis labels readable
    axis.text.y = element_text(size = 12),  # Make y-axis labels readable
    axis.title.x = element_text(size = 14), # Make x-axis title readable
    axis.title.y = element_text(size = 14), # Make y-axis title readable
    legend.title = element_text(size = 13), # Make legend title readable
    legend.text = element_text(size = 11)   # Make legend text readable
  ) +
  labs(
    x = "UMAP Dimension 1",  # Label for x-axis
    y = "UMAP Dimension 2",  # Label for y-axis
    color = "Cell Type",      # Legend title
    title = "UMAP embedding of unintegrated data"  # Title
  )

ggplot(metadata2, aes(x = umapunintegrated_1, y = umapunintegrated_2, color = data)) +
  geom_point(size = 1, alpha = 0.7) +  # Adjust point size and transparency
  scale_color_brewer(palette = "Set1") +  # Use the Set3 color palette
  #scale_color_manual(values = my_colors) +
  theme_classic() +  # Use a clean thmy_coloeme
  theme(
    axis.text.x = element_text(size = 12),  # Make x-axis labels readable
    axis.text.y = element_text(size = 12),  # Make y-axis labels readable
    axis.title.x = element_text(size = 14), # Make x-axis title readable
    axis.title.y = element_text(size = 14), # Make y-axis title readable
    legend.title = element_text(size = 13), # Make legend title readable
    legend.text = element_text(size = 11)   # Make legend text readable
  ) +
  labs(
    x = "UMAP Dimension 1",  # Label for x-axis
    y = "UMAP Dimension 2",  # Label for y-axis
    color = "Assay type",      # Legend title
    title = "UMAP embedding of unintegrated data"  # Title
  )





#varimax_loading_df.to_csv()
#pca_scores_varimax_df_merged.to_csv()




library(ggplot2)
library(RColorBrewer)
color_palette <- brewer.pal(n = 12, name = "Set1")

scores_df_merged = read.csv('/home/delaram/sciRED//review_analysis//pca_scores_varimax_df_3p_5p_PBMC.csv')
loading_df = read.csv('/home/delaram/sciRED//review_analysis/varimax_loading_df_3p_5p_PBMC.csv')

head(scores_df_merged)
head(loading_df)


cond = abs(scores_df_merged$F1)<(70)
cond = abs(scores_df_merged$F4)<(40)
sum(!cond)
scores_df_merged_sub = scores_df_merged[cond,] 

factor_number = 9
ggplot(scores_df_merged_sub, aes(x = F16, y = F9, color = data)) +
  geom_point(size = 1.6, alpha = 0.6) +  
  scale_color_brewer(palette = "Set1") + 
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "F1 Score",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "assay",  # Legend title
    title = paste0("Scatter Plot of F1 vs F", 
                   factor_number,  " colored by assay")
  )


factor_number = 9
ggplot(scores_df_merged_sub, aes(x = F16, y = F9, color = predicted.id)) +
  geom_point(size = 1.6, alpha = 0.6) +  
  scale_color_manual(values = my_colors) +
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "F1 Score",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )



ggplot(scores_df_merged_sub, aes(x = umapunintegrated_1, y = umapunintegrated_2, color = F9)) +
  geom_point(size = 1, alpha = 0.7) +  
  scale_color_gradient(low = "lightyellow2", high = "red3")+
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "Unintegrated UMAP_1",  # Label for x-axis
    y = 'Unintegrated UMAP_2'  # Label for y-axis
    #color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )


ggplot(scores_df_merged_sub, aes(x = umap_1, y = umap_2, color = abs(F4))) +
  geom_point(size = 1, alpha = 0.7) +  
  scale_color_gradient(low = "lightyellow2", high = "red3")+
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "UMAP_1",  # Label for x-axis
    y = 'UMAP_2'  # Label for y-axis
    #color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )


factor_number = 1
ggplot(scores_df_merged_sub, aes(y = abs(F1), x = predicted.id,fill = predicted.id)) +
  geom_boxplot() +  
  scale_fill_manual(values = my_colors) +  
  theme_classic() +  # Use a clean theme
  theme(
    axis.text.x = element_text(size = 12, angle=90),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )



factor_number = 9
ggplot(scores_df_merged_sub, aes(y = F13, x = data,fill = data)) +
  geom_boxplot() +  # Adjust point size and transparency
  scale_fill_brewer(palette = "Set1") +  # Use the custom Set3 color palette
  theme_classic() +  # Use a clean theme
  theme(
    axis.text.x = element_text(size = 16),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )


















###################################
metadata = read.csv('~/sciRED/review_analysis/Ischemia_Reperfusion_Responses_Human_Lung_Transplants_metadata.csv')
head(metadata)
table(metadata$CyclonePhase, metadata$Phase)
table(metadata$tissue)
table(metadata$ltx_case)
table(metadata$sample_name)
table(metadata$sample_name, metadata$ltx_case)
table(metadata$timepoint)
table(metadata$recipient_origin)
table(metadata$donor_id, metadata$sample_name)
table(metadata$development_stage)
table(metadata$tissue)
table(metadata$donor_id)



