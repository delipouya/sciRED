library(RColorBrewer)
library(ggplot2)
library(gridExtra)

load('~/HumanLiver/extra_files/inst/liver/HumanLiver.RData')
load('~/HumanLiver/extra_files/inst/liver/HumanLiver_savedRes.RData')
HumanLiverSeurat = UpdateSeuratObject(HumanLiverSeurat)

##### calculating the mt percentage
#HumanLiverSeurat[["percent.mt"]] <- PercentageFeatureSet(HumanLiverSeurat, pattern = "^MT-")
mt_indices = grep('^MT-',rownames(HumanLiverSeurat))
sum_MT_conuts = colSums(GetAssayData(HumanLiverSeurat, layer = 'counts')[mt_indices,])
sum_counts = colSums(GetAssayData(HumanLiverSeurat, layer = 'counts'))
HumanLiverSeurat[["percent.mt"]]  = sum_MT_conuts/sum_counts

meta_data = HumanLiverSeurat@meta.data
pca_df = Embeddings(HumanLiverSeurat, 'pca')


HumanLiverSeurat <- RunUMAP(HumanLiverSeurat, dims = 1:30, reduction = "pca")
umap_df = Embeddings(HumanLiverSeurat, 'umap')
head(HumanLiverSeurat)
head(umap_df)
sum(colnames(HumanLiverSeurat) != rownames(umap_df))
umap_df2 = cbind(umap_df, HumanLiverSeurat@meta.data)
dim(umap_df2)
dim(umap_df)


library(RColorBrewer)
num_colors = length(names(table(umap_df2$cell_type)))
color_palette <- brewer.pal(n = , name = "Set3")

# Generate a 20-color palette from Set3 (or you can use another palette like Paired)
colors_20 <- brewer.pal(12, "Set3")  # Max 12 colors for Set3, so combine palettes
colors_20 <- c(colors_20, brewer.pal(8, "Dark2"))  # Combine with another palette

# Visualize
barplot(rep(1, 20), col = colors_20, border = NA)

my_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22')
ggplot(umap_df2, aes(x=umap_1, y=umap_2, color=cell_type))+
  geom_point(size = 1.4, alpha = 0.8) +  # Adjust point size and transparency
  #scale_color_brewer(palette = "Set1") +  # Use the Set3 color palette
  scale_color_manual(values = colors_20) +
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
    color = 'Cell type',#"Sample",      # Legend title
    #title = "UMAP embedding of unintegrated data"  # Title
  )

################################################################
################# DO NOT RUN again #####################

################ Saving the object to be used in Python
seur <- CreateSeuratObject(GetAssayData(HumanLiverSeurat, 'counts'))
seur@meta.data = HumanLiverSeurat@meta.data


# The following 13086 features requested have not been scaled
annotations=c('Hep1','abT cell','Hep2','infMac','Hep3','Hep4','plasma cell',
              'NK-like cell','gdT cell1','nonInfMac','periportal LSEC',
              'central venous LSEC','portal Endothelial cell','Hep5','Hep6',
              'mature B cell','cholangiocyte','gdT cell2','erythroid cell',
              'hepatic stellate cell')

label_df = data.frame(cluster=paste0('cluster_',1:20),labels=annotations)
Idents(HumanLiverSeurat) = paste0('cluster_', as.character(HumanLiverSeurat$res.0.8))
human_liver_annot = data.frame(umi=colnames(HumanLiverSeurat), cluster=Idents(HumanLiverSeurat))
human_liver_annot = merge(human_liver_annot, label_df, by.x='cluster', by.y='cluster', all.x=T, sort=F)

human_liver_annot_sorted <- human_liver_annot[match(colnames(HumanLiverSeurat), human_liver_annot$umi),]
sum(human_liver_annot_sorted$umi != colnames(HumanLiverSeurat))
HumanLiverSeurat$cell_type = human_liver_annot_sorted$labels

HumanLiverSeurat$sample = unlist(lapply(strsplit(colnames(HumanLiverSeurat), '_'), '[[', 1))

SaveH5Seurat(seur, filename ='~/sciFA/Data/HumanLiverAtlas.h5Seurat' ,overwrite = TRUE)
Convert('~/sciFA/Data/HumanLiverAtlas.h5Seurat', dest = "h5ad")
################################################################################
################################################################################


tsne_df = Embeddings(HumanLiverSeurat, 'tsne')
tsne_df = cbind(tsne_df, HumanLiverSeurat@meta.data)
##### reading factor
factor_loading = read.csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv')
factor_scores = read.csv('/home/delaram/sciFA/Results/factor_scores_umap_df_humanlivermap.csv')
factor_scores = factor_scores[factor_scores$id %in% row.names(tsne_df),]

sum(factor_scores$id != row.names(tsne_df))
tsne_df_merged = cbind(tsne_df, factor_scores)

n <- 30
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
tsne_df_merged_2 = tsne_df_merged[,-c(45:51)]
#tsne_df_merged_2 = tsne_df_merged


ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=CELL_TYPE))+geom_point(size=1.3)+
  theme_classic()+scale_color_manual(values = col_vector)
ggplot(tsne_df_merged_2, aes(umap_1,umap_2,color=cell_type))+geom_point(size=1)+theme_classic()+scale_color_manual(values = col_vector)

ggplot(tsne_df_merged_2, aes(factor_2,factor_3,color=cell_type))+
  geom_point(size=0.6)+theme_classic()+scale_color_manual(values = col_vector)+xlab('F3')+ylab('F4')

ggplot(tsne_df_merged_2, aes(factor_8,factor_12,color=cell_type))+
  geom_point(size=0.6)+theme_classic()+scale_color_manual(values = col_vector)+xlab('F9')+ylab('F13')

ggplot(tsne_df_merged_2, aes(factor_20,factor_23,color=cell_type))+
  geom_point(size=0.8)+theme_classic()+scale_color_manual(values = col_vector)+xlab('F21')+ylab('F24')

ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=factor_9))+geom_point(size=0.6,alpha=0.6)+
  theme_classic()+scale_color_viridis_b(option='plasma',direction = -1)
ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=factor_18))+geom_point(alpha=0.7)+
  theme_classic()+scale_color_viridis_b(direction = -1)
ggplot(tsne_df_merged_2, aes(umap_1,umap_2,color=factor_29))+geom_point(alpha=0.7)+
  theme_classic()+scale_color_viridis_b(direction = -1)

ggplot(tsne_df_merged_2, aes(umap_1,umap_2,color=factor_11))+geom_point(alpha=0.7)+
  theme_classic()+scale_color_viridis_b(direction = -1)

qc_columns = c('total_counts', 'total_features' , 'percent.mt','S.Score', 'G2M.Score')
factor_cols = paste0('factor_', 0:29)
factor_cols = paste0('factor_',c(0, 9, 18, 19, 21, 25, 27, 28, 29))

c(qc_columns, factor_cols) %in% colnames(tsne_df_merged_2)
tsne_df_merged_3 = tsne_df_merged_2[, colnames(tsne_df_merged_2) %in% c(qc_columns, factor_cols)]
head(tsne_df_merged_3)
colnames(cor(tsne_df_merged_3))
cor_mat = cor(tsne_df_merged_3)[qc_columns, factor_cols]
library(pheatmap)
# make the color pallete
clrsp <- colorRampPalette(c("darkgreen", "white", "purple"))   
clrs <- clrsp(200) 
breaks1 <- seq(-1, 1, length.out = 200)
colnames(cor_mat) = paste0('F',c(0, 9, 18, 19, 21, 25, 27, 28, 29)+1)
rownames(cor_mat)[1:3] = c('Total Counts', 'Total Features', 'MT Fraction')
cor_mat.t = t(cor_mat)
pheatmap(cor_mat.t, cluster_cols = F, breaks = breaks1, color =  clrs, display_numbers = T, 
         cluster_rows = F, fontsize_row = 11, fontsize_col = 12)


###### evaluating loadings
source('~/RatLiver/Codes/Functions.R')
Initialize()
library(gprofiler2)

get_gprofiler_enrich <- function(markers, model_animal_name){
  gostres <- gost(query = markers,
                  ordered_query = TRUE, exclude_iea =TRUE, 
                  sources=c('GO:BP' ,'REAC'),
                  organism = model_animal_name)
  return(gostres)
}

factor_loading = read.csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv')
genes = read.csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv')
df = data.frame(gene= genes$X0,factor=factor_loading$X29)
model_animal_name ='hsapiens'

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
enrich_res_pos = data.frame(enrich_res$result)
enrich_res_pos = enrich_res_pos[,!colnames(enrich_res_pos)%in%'evidence_codes']
enrich_res_pos$log_p = -log(as.numeric(enrich_res_pos$p_value))
enrich_res_pos = enrich_res_pos[order(enrich_res_pos$log_p, decreasing = T),]
View(data.frame(1:nrow(enrich_res_pos),enrich_res_pos$term_name, enrich_res_pos$intersection))

#enrich_res_pos = enrich_res_pos[c(1,3,4,5,8,9,11,14,16,19,20,22,27,29,38,41,45,47),]## positive - F29 (=F30) - human liver dataset
enrich_res_pos = enrich_res_pos[c(1,3,4,5,8,9,11,14,16,19,20,22,29,45,47),] ## positive - F29 (=F30) - human liver dataset
#enrich_res_pos = enrich_res_pos[c(1,2,3,6,7,8,13,15,17,18,30),]## positive - F9 (=F10) - human liver dataset
#enrich_res_pos = enrich_res_pos[c(2,5,8,10,12,14,15,19,20,46,54),]## positive - F18 (=F19) - human liver dataset
#enrich_res_pos = enrich_res_pos[c(1, 3, 4, 6, 7, 11, 12, 15,20),]## positive - F28 (=F29) - human liver dataset

enrich_res_pos = enrich_res_pos[,colnames(enrich_res_pos) %in% c('term_name', 'log_p')]

enrich_res_pos$term_name = gsub('metabolic process', 'metabolism',enrich_res_pos$term_name)
enrich_res_pos$term_name = gsub('Binding and ', '',enrich_res_pos$term_name)
enrich_res_pos$term_name = gsub('Classical ', '',enrich_res_pos$term_name)
enrich_res_pos$term_name = gsub('endoplasmic reticulum ', 'ER ',enrich_res_pos$term_name)
enrich_res_pos$term_name = gsub('cellular ', '',enrich_res_pos$term_name)
enrich_res_pos$term_name
#enrich_res_pos$term_name[16] = 'FCGR dependent phagocytosis' ## positive - F29 (=F30) - human liver dataset
enrich_res_pos$term_name

enrich_res_pos$term_name <- factor(enrich_res_pos$term_name, levels =  enrich_res_pos$term_name[length(enrich_res_pos$term_name):1])

title = ''#'stim'#'Male'
fill_color='#80B1D3'
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity',fill=fill_color,color='grey3')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  scale_fill_manual(values = c(fill_color))+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 14, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))

c("#1F78B4","#7570B3",'#B15928', '#F0027F')
c('#80B1D3','#DECBE4',"#FDC086", "#E78AC3")


# start from: 0 ---> factor_9, factor_18, factor_29
sum(!factor_scores$factor_9 == tsne_df_merged_2$factor_9)
sum(!factor_scores$factor_18 == tsne_df_merged_2$factor_18)

factor_scores$CELL_TYPE = factor_scores$cell_type
table(factor_scores$cell_type)
factor_scores$CELL_TYPE = gsub('central venous ', 'cv', factor_scores$CELL_TYPE)
factor_scores$CELL_TYPE = gsub('periportal ', 'pp', factor_scores$CELL_TYPE)
factor_scores$CELL_TYPE[factor_scores$CELL_TYPE=='portal Endothelial cell'] = 'pEndothelial'
factor_scores$CELL_TYPE = gsub('hepatic ', '', factor_scores$CELL_TYPE)

table(factor_scores$CELL_TYPE)
head(factor_scores)

ggplot(factor_scores, aes(x=CELL_TYPE, y=factor_26, fill=CELL_TYPE))+geom_boxplot()+
  scale_fill_manual(values = col_vector)+theme_classic()+xlab('')+ylab('F10 Score')+
  theme(axis.text.x = element_text(size = 13.5, angle = 90), axis.text.y = element_text(size = 15),
        axis.title.y = element_text(size = 15))

ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=factor_26))+geom_point(alpha=0.9,size=1.3)+
  theme_classic()+scale_color_viridis_b(name = "Factor-29\nScore",direction = +1)+
  theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14),
        axis.title.y = element_text(size = 16), axis.title.x = element_text(size = 16))
  
ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=CELL_TYPE))+geom_point(size=3)+
  theme_classic()+scale_color_manual(values = col_vector)+
  theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14),
        axis.title.y = element_text(size = 16), axis.title.x = element_text(size = 16))

ggplot(tsne_df_merged_2, aes(umap_1,umap_2,color=factor_28))+geom_point(alpha=0.7,size=0.8)+
  theme_classic()+scale_color_viridis_b(direction = +1)



###########################################################################
######################## visualizing factor loadings 
###########################################################################
factor_loading = read.csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv')
genes = read.csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv')
colnames(factor_loading) = paste0('F', 1:ncol(factor_loading))
df = data.frame(genes= genes$X0,factor=factor_loading$F29)

varimax_loading_df_ord = df[order(df$factor, decreasing = F),]
varimax_loading_vis = head(varimax_loading_df_ord, 20)
varimax_loading_vis$genes
varimax_loading_vis$genes = gsub('-ENS.*', '',varimax_loading_vis$genes)
varimax_loading_vis$genes
varimax_loading_vis = varimax_loading_vis[order(varimax_loading_vis$factor, decreasing = F),]
varimax_loading_vis$genes <- factor(varimax_loading_vis$genes, levels=varimax_loading_vis$genes)



col_vis = '#F0027F'#"#7570B3"#"#1F78B4"
ggplot(varimax_loading_vis, aes(y=factor, x=genes, color=factor))+geom_point(size=3)+theme_bw()+
  theme(axis.text.x = element_text(color = "grey20", size = 11, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 10, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 15, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(hjust = 1,angle = 0),
        legend.position="right", legend.direction="vertical")+
  scale_color_gradient()+
  #scale_colour_gradientn(colours=c("red", "blue"))+
  scale_color_gradient2(name='',midpoint = 0, low = col_vis, mid = "white",
                        high = "white", space = "Lab" )+
  ylab('Loading')+xlab('')
#humanliver_f30_dotplot_v2

c("#1F78B4","#7570B3",'#B15928', '#F0027F')
c('#80B1D3')


genes = read.csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv')
df = data.frame(gene= genes$X0,factor=factor_loading$X28)
df_pos = df[order(df$factor, decreasing = T),]
df_pos = df[order(df$factor, decreasing = F),]

markers_vis = head(df_pos,10)
dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(markers_vis, theme=tt2)
gridExtra::grid.table(markers_vis)
