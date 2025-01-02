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




###########################################################################################
#######################. evaluate correlation with QC parameters. ########################
###########################################################################################
Kang18_8vs8_seur <- readRDS("~/scLMM/LMM-scRNAseq//Data/PBMC_Lupus_Kang8vs8_data_counts.rds")
factor_df = read.csv('~/sciFA/Results/pca_scores_varimax_df_merged_lupusPBMC.csv')
head(factor_df)
##### calculating the mt percentage
#HumanLiverSeurat[["percent.mt"]] <- PercentageFeatureSet(HumanLiverSeurat, pattern = "^MT-")
mt_indices = grep('^MT-',rownames(Kang18_8vs8_seur))
sum_MT_conuts = colSums(GetAssayData(Kang18_8vs8_seur, layer = 'counts')[mt_indices,])
sum_counts = colSums(GetAssayData(Kang18_8vs8_seur, layer = 'counts'))
Kang18_8vs8_seur[["percent.mt"]]  = sum_MT_conuts/sum_counts

sum(colnames(Kang18_8vs8_seur)!=factor_df$X)
factor_df$percent.mt = Kang18_8vs8_seur[["percent.mt"]] 

qc_columns = c('nCount_originalexp', 'nFeature_originalexp' )
factor_cols = paste0('F', c(1, 3, 4, 8, 13, 15, 16, 17, 19, 21:24, 26:30 ))
to_keep_cols = c(factor_cols, qc_columns)
factor_df_sub = factor_df[,colnames(factor_df) %in% to_keep_cols]

cor_mat = cor(factor_df_sub)[qc_columns, factor_cols]


library(pheatmap)
# make the color pallete
clrsp <- colorRampPalette(c("darkgreen", "white", "purple"))   
clrs <- clrsp(200) 
breaks1 <- seq(-1, 1, length.out = 200)
rownames(cor_mat)[1:2] = c('Total Counts', 'Total Features')
cor_mat.t = t(cor_mat)
pheatmap(cor_mat.t, cluster_cols = F, breaks = breaks1, color =  clrs, display_numbers = T, 
         cluster_rows = F, fontsize_row = 11, fontsize_col = 12)



factor_number=25
factor_df2 = factor_df
factor_df2 = factor_df2[!is.na(factor_df2$cell),]
#factor_df2 = factor_df2[factor_df2$F22<(20),]


ggplot(factor_df2, aes(y = F29, x = cell,fill = cell)) +
  geom_boxplot() +  
  scale_fill_manual(values = color_palette) +  
  theme_classic() +  # Use a clean theme
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) + coord_flip()+
  labs(
    x = "",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )




###################################################################

library(gprofiler2)

get_gprofiler_enrich <- function(markers, model_animal_name){
  gostres <- gost(query = markers,
                  ordered_query = TRUE, exclude_iea =TRUE, 
                  sources=c('GO:BP' ,'REAC'),
                  organism = model_animal_name)
  return(gostres)
}

factor_loading = read.csv('~/sciFA/Results/varimax_loading_df_lupusPBMC.csv')

factor_i = 3
df = data.frame(gene = factor_loading$X, factor = factor_loading[[paste0('F',factor_i)]])
df$gene <- sapply(strsplit(as.character(df$gene), "-"), `[`, 1)
df$gene <- sub("-EN.*", "", df$gene)
# Select top and bottom 20 genes
varimax_loading_vis = rbind(head(df[order(df$factor, decreasing = TRUE), ], 20), 
                            tail(df[order(df$factor, decreasing = TRUE), ], 20))

# Factor levels based on the ordering of genes
varimax_loading_vis$gene <- factor(varimax_loading_vis$gene, 
                                   levels = varimax_loading_vis$gene)

# Plot with vertical orientation using coord_flip()
ggplot(varimax_loading_vis, aes(x = gene, y = factor, color = factor)) +
  geom_point(size = 2, alpha = 1) +
  theme_bw() +
  theme(
    axis.text.x = element_text(color = "grey20", size = 12, angle=-90),  # Adjust font size for horizontal axis
    axis.text.y = element_text(color = "grey20", size = 11.5, hjust = 1, vjust = 0.5),
    axis.title.x = element_text(color = "grey20", size = 14),
    axis.title.y = element_text(color = "grey20", size = 14),
    legend.text = element_text(hjust = 1),
    legend.position = "left",
    legend.direction = "vertical"
  ) +
  # Set gradient colors
  scale_color_gradient2(
    name = paste0("Factor ", factor_i),
    midpoint = 0,
    low = "darkgreen",   # Color for negative values
    mid = "white",       # Color for midpoint
    high = "darkred",    # Color for positive values
    space = "Lab"
  ) +
  ylab('Factor loading') +
  xlab('') +
  ggtitle(paste0("Factor ", factor_i))
  #coord_flip()  # Flip the coordinates to make the plot vertical





model_animal_name ='hsapiens'


factor_i = 22
df = data.frame(gene = factor_loading$X, factor = factor_loading[[paste0('F',factor_i)]])
#df$gene <- sapply(strsplit(as.character(df$gene), "-"), `[`, 1)
df$gene <- sub("-EN.*", "", df$gene)


df_pos = df[order(df$factor, decreasing = T),]
df_neg = df[order(df$factor, decreasing = F),]

head(df_pos,10)
head(df_neg,10)
num_genes = 200

table_to_vis = df_pos[1:20,]
rownames(table_to_vis) = NULL
colnames(table_to_vis) = c('Gene', 'Score')
table_to_vis$Score = round(table_to_vis$Score, 3)
library(gridExtra)
dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(table_to_vis, theme=tt2)

######## pos enrichment
enrich_res = get_gprofiler_enrich(markers=df_pos$gene[1:num_genes], 
                                  model_animal_name = model_animal_name )#'gp__SEA8_T0ld_VHU'
######## neg enrichment
enrich_res = get_gprofiler_enrich(markers=df_neg$gene[1:num_genes], 
                                  model_animal_name = model_animal_name )#'gp__SEA8_T0ld_VHU'
head(enrich_res$result,30)



enrich_res_df = data.frame(enrich_res$result)
enrich_res_df$log_p = -log(as.numeric(enrich_res_df$p_value))
enrich_res_df = enrich_res_df[order(enrich_res_df$log_p, decreasing = T),]
#View(enrich_res_df)

num_term_vis = 15
enrich_res_df = enrich_res_df[1:num_term_vis,]
#enrich_res_df = enrich_res_df[c(2, 5,28,29,32,42,46,48,62,65),]

enrich_res_df = enrich_res_df[,colnames(enrich_res_df) %in% c('term_name', 'p_value')]
enrich_res_df$log_p = -log(enrich_res_df$p_value)
title = ''

enrich_res_df$term_name = gsub('metabolic process', 'metabolism',enrich_res_df$term_name)
#enrich_res_df$term_name[9] = "The citric acid (TCA) cycle"
enrich_res_df$term_name <- factor(enrich_res_df$term_name, 
                                   levels =  enrich_res_df$term_name[length(enrich_res_df$term_name):1])

color = "coral3"
color = "darkseagreen3"

title = ''#'stim'#'Male'
enrich_res_df = enrich_res_df[-7,]
ggplot(enrich_res_df, aes(y=term_name,x=log_p))+
  geom_bar(stat = 'identity',fill=color,color='grey10')+
  xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  scale_fill_manual(values = c(color))+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 12, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))



