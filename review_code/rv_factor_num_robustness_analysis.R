numcomp_files = list.files(path = '/home/delaram/sciRED/review_analysis/factor_robustness/', 
           pattern = 'humanliver_varimax_scores_numcomp*', full.names = T)

numcomp_list = lapply(numcomp_files, read.csv)
names(numcomp_list) = c('numcomp_10', 'numcomp_20', 'numcomp_30', 'numcomp_5', 'numcomp_50')
lapply(numcomp_list, head)
lapply(numcomp_list, dim)
numcomp_list_factors = lapply(numcomp_list, function(x) x[,11:ncol(x)])
lapply(numcomp_list_factors, head)
lapply(numcomp_list_factors, dim)


# Load required packages
library(ggplot2)
library(reshape2)

# Step 1: Function to calculate pairwise correlations between all factors of two matrices
pairwise_factor_cor <- function(mat1, mat2) {
  # Get the number of factors in both matrices
  num_factors1 <- ncol(mat1)
  num_factors2 <- ncol(mat2)
  
  # Initialize an empty matrix to store the correlations
  cor_matrix <- matrix(0, num_factors1, num_factors2, 
                       dimnames = list(paste0("Factor", 1:num_factors1),
                                       paste0("Factor", 1:num_factors2)))
  
  # Step 2: Calculate pairwise correlations between factors
  for (i in 1:num_factors1) {
    for (j in 1:num_factors2) {
      cor_matrix[i, j] <- cor(mat1[, i], mat2[, j])
    }
  }
  
  return(cor_matrix)
}

# Step 3: Loop over all pairwise combinations of matrices in `numcomp_list_factors`
matrix_names <- names(numcomp_list_factors)
n <- length(matrix_names)

# Initialize a list to store pairwise correlation results
pairwise_cor_list <- list()

for (i in 1:(n-1)) {
  for (j in (i+1):n) {
    mat1 <- numcomp_list_factors[[i]]
    mat2 <- numcomp_list_factors[[j]]
    
    # Calculate pairwise correlation matrix between the two matrices
    cor_matrix <- pairwise_factor_cor(mat1, mat2)
    
    # Store the result in the list
    pairwise_cor_list[[paste(matrix_names[i], matrix_names[j], sep = "_vs_")]] <- cor_matrix
  }
}

# Step 4: Visualize pairwise correlations for each combination using heatmap
plot_heatmap <- function(cor_matrix, title) {
  # Convert the correlation matrix into long format for ggplot2
  cor_long <- melt(cor_matrix)
  
  # Plot the heatmap
  ggplot(cor_long, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0,
                         limit = c(-1, 1), space = "Lab", 
                         name = "Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          axis.text.y = element_text(size = 10)) +
    labs(title = title, x = "", y = "")
}

plot_heatmap_abs <- function(cor_matrix, title) {
  # Convert the correlation matrix into long format for ggplot2
  cor_long <- melt(abs(cor_matrix))  # We are using absolute correlations
  
  # Plot the heatmap with refined color scheme
  ggplot(cor_long, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "darkgreen",
                        limit = c(0, 1), name = "Absolute Correlation") + 
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          axis.text.y = element_text(size = 10)) +
    labs(title = title, x = "", y = "")
}

i = 10
length(pairwise_cor_list)
pair_name = names(pairwise_cor_list)[i]
plot_heatmap_abs(pairwise_cor_list[[pair_name]], title = paste("Correlation between\n", pair_name))
plot_heatmap(pairwise_cor_list[[pair_name]], title = paste("Correlation between\n", pair_name))



# Define the directory path
output_dir <- "/home/delaram/sciRED/review_analysis/factor_robustness/humanliver/"
# Define the directory path
output_dir <- "/home/delaram/sciRED/review_analysis/factor_robustness/humanliver/"

# Function to save both heatmaps (absolute and regular)
save_heatmaps <- function(pair_name) {
  cor_matrix <- pairwise_cor_list[[pair_name]]
  
  # Get dimensions of the correlation matrix
  num_factors <- ncol(cor_matrix)
  
  # Set width and height based on the number of factors
  width <- 4 + num_factors * 0.5  # Example formula for width
  height <- 4 + num_factors * 0.3  # Example formula for height
  
  # Define file names for each heatmap
  abs_filename <- paste0(output_dir, "heatmap_abs_", pair_name, ".png")
  regular_filename <- paste0(output_dir, "heatmap_", pair_name, ".png")
  
  # Generate and save absolute correlation heatmap
  heatmap_abs_plot <- plot_heatmap_abs(cor_matrix, 
                                       title = paste("Correlation between\n", pair_name))
  ggsave(filename = abs_filename, plot = heatmap_abs_plot, width = width, height = height)
  
  # Generate and save regular correlation heatmap
  heatmap_plot <- plot_heatmap(cor_matrix, 
                               title = paste("Correlation between\n", pair_name))
  ggsave(filename = regular_filename, plot = heatmap_plot, width = width, height = height)
}

# Loop through all pair names and save the heatmaps
for (pair_name in names(pairwise_cor_list)) {
  save_heatmaps(pair_name)
}


