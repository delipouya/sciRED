
library(ggplot2)
library(reshape2)
library(pheatmap)

data_HVG = read.csv('/home/delaram//sciRED/review_analysis/pca_scores_varimax_df_merged_lupusPBMC_numgenes_2000.csv')
data_all = read.csv('/home/delaram//sciRED/review_analysis/pca_scores_varimax_df_merged_lupusPBMC_numgenes_18890.csv')

head(data_HVG)
head(data_all)



# Assuming pca_scores_varimax_df and pca_scores_varimax_df_lib are your data frames
# Remove the first column if it's an index
df1 <- data_HVG[,10:ncol(data_HVG)]
df2 <- data_all[,10:ncol(data_all)]

df1[,1]==df2[,1]
# Combine the two data frames
### add lib to colnames of df2
colnames(df1) = paste0(colnames(df1), '_hvg')
merged_df <- cbind(df1, df2)

# Calculate the pairwise correlations
correlation_matrix <- cor(merged_df)
# Extract the diagonal of the correlation matrix
diagonal_values <- diag(correlation_matrix)
# Print the diagonal values
print(diagonal_values)
head(correlation_matrix)

correlation_matrix = correlation_matrix[1:30, 31:ncol(correlation_matrix)] #x: lib+sample - y:lib
rownames(correlation_matrix)
colnames(correlation_matrix)
# Option 1: Visualize with pheatmap
pheatmap(correlation_matrix, 
         color = colorRampPalette(c("blue", "white", "red"))(50),
         main = "Pairwise Correlation Between Factors from\nPBMC dataset with all genes and HVGs alone", 
         cluster_rows = F, cluster_cols = F)

# Option 2: Visualize with ggplot2
# Melt the correlation matrix for ggplot2
melted_cormat <- melt(correlation_matrix)
# Create the heatmap
melted_cormat$Var1 = gsub(pattern = '_hvg', '', melted_cormat$Var1)
ggplot(data = melted_cormat, aes(Var1, Var2, fill = value)) +
  geom_tile(colour = "black") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", name="Correlation") +
  #theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  ggtitle("Pairwise Correlation Between Factors from\nPBMC dataset with all genes and HVGs alone")+
  ylab('All genes included')+xlab('HVGs')

melted_cormat2 = melted_cormat
melted_cormat2$value = abs(melted_cormat$value)


# Define the correct order for factors F1 to F30
factor_order <- paste0("F", 1:30)

# Convert Var1 and Var2 to factors with the specified order
melted_cormat2$Var1 <- factor(melted_cormat2$Var1, levels = factor_order)
melted_cormat2$Var2 <- factor(melted_cormat2$Var2, levels = factor_order)

# Plot
ggplot(data = melted_cormat2, aes(Var1, Var2, fill = value)) +
  geom_tile(colour = "black") +
  scale_fill_gradient2(low = "white", high = "darkgreen", mid='darkseagreen2',
                       midpoint = 0.5, limit = c(0, 1), space = "Lab", name="Absolute\nCorrelation") +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1, size = 13),
        axis.text.y = element_text(size = 13)) +
  ggtitle("Pairwise Correlation Between Factors from\nPBMC dataset with all genes and HVGs alone") +
  ylab('All genes included') + 
  xlab('HVGs')

library(dplyr)


corr_thr = 0.6
# Assuming 'melted_cormat2' is your melted correlation matrix with columns Var1, Var2, and value
# Filter for correlations greater than 0.8
filtered_matches <- melted_cormat2 %>%
  filter(value > corr_thr) %>%
  group_by(Var1) %>%  # Group by factor in Var1
  slice_max(order_by = value, n = 1) %>%  # Keep the highest correlation per Var1
  ungroup()

# Count the number of matches
num_matches <- nrow(filtered_matches)

# View the filtered matches
filtered_matches
filtered_matches = as.data.frame(filtered_matches)
colnames(filtered_matches) = c('HVG', 'All', 'Corr')
filtered_matches$Corr = round(filtered_matches$Corr, 3)
dev.off()
gridExtra::grid.table(filtered_matches)

# Output the number of matches
num_matches
