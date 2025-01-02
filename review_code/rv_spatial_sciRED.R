library("spatialLIBD")
library(optparse)
library(SingleCellExperiment)
library("zellkonverter")
library(SeuratData)
library(SeuratDisk)
library(Matrix)

######################################################
sce_path_zip <- spatialLIBD::fetch_data("spatialDLPFC_snRNAseq")
sce_path <- unzip(sce_path_zip, exdir = tempdir())
sce <- HDF5Array::loadHDF5SummarizedExperiment(
  file.path(tempdir(), "sce_DLPFC_annotated")
)

dim(sce)
meta_data = colData(sce)
table(meta_data$Sample)
table(meta_data$SAMPLE_ID)
table(meta_data$SAMPLE_ID, meta_data$Sample)
table(meta_data$age)
table(meta_data$sex)
table(meta_data$diagnosis)
table(meta_data$layer_annotation)

table(meta_data$cellType_layer)
table(meta_data$cellType_layer, meta_data$layer_annotation)

table(meta_data$cellType_hc)
table(meta_data$cellType_k)

table(meta_data$cellType_broad_hc)
table(meta_data$cellType_broad_k)
table(meta_data$cellType_broad_hc, meta_data$cellType_broad_k)

table(meta_data$BrNum)
hist(meta_data$subsets_Mito_percent)
hist(meta_data$sum)
hist(meta_data$subsets_Mito_sum)
hist(meta_data$total)
table(meta_data$kmeans)

mat <- counts(sce)
dim(mat)

################age######################################################
# sce for the SingleCellExperiment object containing the spot-level data that includes the
#information for visualizing the clusters/genes on top of the Visium histology,
sce <- spatialLIBD::fetch_data(type="sce")
table(sce$sample_name)
mat <- counts(sce)
colData(sce)
sce <- scuttle::addPerCellQC(sce)
colnames(sce) = paste0(colnames(sce),'_' ,sce$sample_name)
length(colnames(sce))
length(unique(colnames(sce)))
sce_seur = as.Seurat(sce)

length(row.names(sce))
length(unique(row.names(sce)))
metadata = as.data.frame(colData(sce))
head(metadata)

######################################################
sparse_mat = GetAssayData(sce_seur)

write.csv(metadata, "~/sciRED/review_analysis/spatial_Dorsolateralcortext_metadata.csv")
# Save the row names (gene names) and column names (cell IDs) separately as CSV files
write.csv(rownames(sparse_mat), "~/sciRED/review_analysis/spatial_Dorsolateralcortext_row_names.csv", row.names = FALSE)
write.csv(colnames(sparse_mat), "~/sciRED/review_analysis/spatial_Dorsolateralcortext_col_names.csv", row.names = FALSE)

metadata = read.csv("~/sciRED/review_analysis/spatial_Dorsolateralcortext_metadata.csv")
table(metadata$sample_name, metadata$replicate)
table(metadata$sample_name, metadata$subject)

######################################################

plot1 <- VlnPlot(sce_seur, features = "nCount_originalexp", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(sce_seur, features = "spatialLIBD") + theme(legend.position = "right")
wrap_plots(plot1, plot2)


######################################################
################# writing the data into a file using npz and scipy directly
library(reticulate)
# Convert sparse matrix to Python object and save as .npz
scipy <- import("scipy.sparse")
np <- import("numpy")

# Save the sparse matrix in .npz format (from earlier step)
scipy$save_npz(path.expand("~/sciRED/review_analysis/spatial_Dorsolateralcortext_matrix.npz"), sparse_mat)
# Save the sparse matrix to .npz format
scipy$save_npz(file_path, sparse_mat)


######################################################
################# writing the data into a file using mtx
writeMM(sparse_mat, "~/sciRED/review_analysis/spatial_Dorsolateralcortext_matrix.mtx")

######################################################
################# writing the data into a file using h5ad 

SaveH5Seurat(sce_seur, filename = "~/sciRED/review_analysis/sce_spatial.h5Seurat")
Convert("~/sciRED/review_analysis/sce_spatial.h5Seurat", dest = "h5ad")

writeH5AD(
  sce,
  file='~/sciRED/review_analysis/sce_spatial.h5ad',
  X_name = NULL,
  skip_assays = FALSE,
  compression = c("none", "gzip", "lzf"),
  verbose = NULL)



######################################################
### save the varimax_loading_df and varimax_scores to a csv file
varimax_loading_df = read.csv('/home/delaram/sciRED//review_analysis//varimax_loading_df_spatial_Dorsolateralcortext.csv')
pca_scores_varimax_df = read.csv('/home/delaram/sciRED//review_analysis//pca_scores_varimax_df_spatial_Dorsolateralcortext.csv')


### load a subject's factors
varimax_loading_df_lib = read.csv('/home/delaram/sciRED//review_analysis/spatial//varimax_loading_df_spatial_Dorsolateralcortext_Br5292.csv') #Br5292, Br5595, Br8100
pca_scores_varimax_df_lib = read.csv('/home/delaram/sciRED//review_analysis/spatial//pca_scores_varimax_df_spatial_Dorsolateralcortext_Br5292.csv')


### load a subject's factors
varimax_loading_df = read.csv('/home/delaram/sciRED//review_analysis/spatial//varimax_loading_df_spatial_Dorsolateralcortext_Br5292_reg_libsubj.csv') #Br5292, Br5595, Br8100
pca_scores_varimax_df = read.csv('/home/delaram/sciRED//review_analysis/spatial//pca_scores_varimax_df_spatial_Dorsolateralcortext_Br5292_reg_libsubj.csv')
head(varimax_loading_df)
head(pca_scores_varimax_df)
dim(pca_scores_varimax_df)


head(pca_scores_varimax_df)
head(pca_scores_varimax_df_lib)

library(ggplot2)
library(reshape2)
library(pheatmap)

# Assuming pca_scores_varimax_df and pca_scores_varimax_df_lib are your data frames
# Remove the first column if it's an index
df1 <- pca_scores_varimax_df[,-1]
df2 <- pca_scores_varimax_df_lib[,-1]

df1[,1]==df2[,1]
# Combine the two data frames
### add lib to colnames of df2
colnames(df2) = paste0(colnames(df2), '_l')
merged_df <- cbind(df1, df2)

# Calculate the pairwise correlations
correlation_matrix <- cor(merged_df)
# Extract the diagonal of the correlation matrix
diagonal_values <- diag(correlation_matrix)
# Print the diagonal values
print(diagonal_values)

correlation_matrix = correlation_matrix[1:30, 31:ncol(correlation_matrix)] #x: lib+sample - y:lib
rownames(correlation_matrix)
colnames(correlation_matrix)
# Option 1: Visualize with pheatmap
pheatmap(correlation_matrix, 
         color = colorRampPalette(c("blue", "white", "red"))(50),
         main = "Pairwise Correlation Between Factors from\nlibrary remove and library+sample removed", 
         cluster_rows = F, cluster_cols = F)

# Option 2: Visualize with ggplot2
# Melt the correlation matrix for ggplot2
melted_cormat <- melt(correlation_matrix)
# Create the heatmap
ggplot(data = melted_cormat, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", name="Correlation") +
  #theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  ggtitle("Pairwise Correlation Between Factors from\nlibrary remove and library+sample removed")+
  xlab('library+sample removed')+ylab('library removed')



metadata = read.csv("~/sciRED/review_analysis/spatial/spatial_Dorsolateralcortext_metadata.csv")
head(metadata)
dim(metadata)
subjects = names(table(metadata$subject))

i = 3
a_subject = subjects[i]
a_subject
table(metadata$sample_name, metadata$replicate)
table(metadata$sample_name, metadata$subject)
metadata = metadata[metadata$subject%in%a_subject,]

######################################################
head(pca_scores_varimax_df)
head(metadata)
sum(pca_scores_varimax_df$X0 != metadata$X)
merged_spatial_factors = cbind(metadata, pca_scores_varimax_df)
library(RColorBrewer)  # For nice categorical color palettes
library(ggplot2)
library(viridis)  # For high-contrast color scales

sample_names = names(table(merged_spatial_factors$sample_name))
sample_names
#c("151507", "151508", "151509", "151510", "151669", "151670", 
#                 "151671", "151672", "151673", "151674", "151675", "151676")
length(sample_names)
i = 1
df <- merged_spatial_factors[merged_spatial_factors$sample_name==sample_names[i],]

# Assuming the actual column names are "imagerow" and "imagecol" for spatial coordinates
df$x_coord <- df$imagerow
df$y_coord <- df$imagecol

fixed_colors <- c("L1" = "#E41A1C",   # Red
                  "L2" = "#377EB8",   # Blue
                  "L3" = "#4DAF4A",   # Green
                  "L4" = "#984EA3",   # Purple
                  "L5" = "#FF7F00",   # Orange
                  "L6" = "#FFFF33",   # Yellow
                  "WM" = "#A65628",   # Brown
                  "NA" = "#999999")   # Grey for 'NA'


# Ensure 'NA' is replaced and layer factor levels are set
df$spatialLIBD[is.na(df$spatialLIBD)] <- 'NA'
df$layer <- factor(df$spatialLIBD, levels = c("L1", "L2", "L3", "L4", "L5", "L6", "WM", "NA"))

# Plot the spatial coordinates with the categorical covariate overlaid
ggplot(df, aes(x = x_coord, y = y_coord, color = layer)) +
  geom_point(alpha = 0.7) +  # Scatter plot of spots
  scale_color_manual(values = fixed_colors) +  # Apply fixed color mapping
  labs(title = paste0(a_subject, ' - sample#', sample_names[i], ' - spatial DLPFC'),
       x = "X Coordinate",
       y = "Y Coordinate") +
  theme_minimal() +
  coord_fixed() +  # Ensures aspect ratio is 1:1
  theme(
    legend.title = element_text(size = 14),  # Increase legend title font size
    legend.text = element_text(size = 12),   # Increase legend item font size
    axis.text = element_text(size = 12),     # Increase axis tick font size
    axis.title = element_text(size = 14),    # Increase axis title font size
    plot.title = element_text(size = 16)     # Increase plot title font size
  )


factor_number = 'F10'
ggplot(df, aes(x = x_coord, y = y_coord, color = F10)) +
  geom_point() +  # Adjust alpha for clearer colors
  scale_color_viridis(option = "plasma", direction = 1) +  # High contrast palette
  labs(title = paste0('sciRED ', factor_number, ' - ',a_subject, ' - sample#',sample_names[i]),
       x = "X Coordinate",
       y = "Y Coordinate") +
  theme_minimal() +
  coord_fixed()+  # Ensures aspect ratio is 1:1
  theme(
    legend.title = element_text(size = 14),  # Increase legend title font size
    legend.text = element_text(size = 12),   # Increase legend item font size
    axis.text = element_text(size = 12),     # Increase axis tick font size
    axis.title = element_text(size = 14),    # Increase axis title font size
    plot.title = element_text(size = 16)  # Increase plot title font size
  )
  

##### boxblot for a subject - all four samples included
df = merged_spatial_factors
# Ensure 'NA' is replaced and layer factor levels are set
df$spatialLIBD[is.na(df$spatialLIBD)] <- 'NA'
df$layer <- factor(df$spatialLIBD, levels = c("L1", "L2", "L3", "L4", "L5", "L6", "WM", "NA"))

factor_number = 'F2'
ggplot(df, aes(x = layer, y = F2, fill = layer)) + 
  geom_boxplot() +  # Create boxplot
  scale_fill_manual(values = fixed_colors) +  # Use the same categorical color palette
  labs(title = paste0('sciRED ', factor_number, ' - sample#',sample_names[i]),
       x = "Layer",
       y = "Factor score") +
  theme_minimal() +
  theme(
    legend.position = "none",  # Remove legend for fill
    axis.text = element_text(size = 14),  # Increase axis tick font size
    axis.title = element_text(size = 14),  # Increase axis title font size
    plot.title = element_text(size = 16)  # Increase plot title font size
  )




