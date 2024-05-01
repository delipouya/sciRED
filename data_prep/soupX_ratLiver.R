source('Codes/Functions.R')
source('Codes/convert_human_to_ortholog_functions.R')
Initialize()

######### importing the samples raw data
seur_DA_1 <- CreateSeuratObject(counts=Read10X('Data/rat_DA_01_reseq/', gene.column = 2), min.cells=0,min.features=1, project = "snRNAseq")
seur_DA_2 = CreateSeuratObject(counts=Read10X('Data/rat_DA_M_10WK_003/', gene.column = 2), min.cells=0,min.features=1, project = "snRNAseq")
seur_LEW_1 <- CreateSeuratObject(counts=Read10X('Data/rat_Lew_01/', gene.column = 2), min.cells=0,min.features=1, project = "snRNAseq")
seur_LEW_2 <- CreateSeuratObject(counts=Read10X('Data/rat_Lew_02/', gene.column = 2), min.cells=0,min.features=1, project = "snRNAseq")

seur_merged = merge(seur_DA_1, c(seur_DA_2, seur_LEW_1, seur_LEW_2), 
                    add.cell.ids = c('rat_DA_01_reseq', 'rat_DA_M_10WK_003', 'rat_Lew_01', 'rat_Lew_02'), 
                    project = "rat_data", merge.data = TRUE)
###########################

#### all the cell names in the merged_samples (after filter) need to be included in the seur_merged object
sum(!colnames(merged_samples) %in% colnames(seur_merged))
seur_mergedSub = seur_merged[,colnames(seur_merged) %in% selected_UMIs]

### need to re-do the lewis-1 as well until saveing the out object
sample_name = 'rat_Lew_01'
file_name = 'rat_Lew_01_mito_40_lib_2000'
input_name = paste0('Results/preproc_rats/', file_name, '.rds')
output_name =  paste0('Results/preproc_rats/', file_name, '_decontam.rds')
seur <- readRDS(input_name)

total_cell_markers <- readRDS('~/XSpecies/Data/Total_markers_converted_df.rds')
hep_markes <- total_cell_markers$Hepatocytes$rnorvegicus_homolog_associated_gene_name
t_cell_markers = total_cell_markers$T_cells$rnorvegicus_homolog_associated_gene_name
mapper = read.delim('~/XSpecies/Data/rat_DA/features.tsv.gz',header=F)

tod = Seurat::Read10X(paste0('Data/',sample_name,'/raw_feature_bc_matrix/'))
toc = seur 
cells_to_keep = colnames(seur)

### profiling the Soup rowSums(toc)/sum(toc)
sc = SoupChannel(tod=tod,toc=tod[,cells_to_keep]) ## i think this only needs the cell names
head(sc$soupProfile)
dim(sc$metaData)

# Picking soup specific genees > are highly expressed in the background
head(sc$soupProfile[order(sc$soupProfile$est, decreasing = TRUE), ], n = 30)
saveRDS(sc$soupProfile,paste0('~/XSpecies/Results/', sample_name,'/',sample_name, '_SoupProfile.rds'))

PC_NUMBER = 30
if(sample_name=='rat_DA_M_10WK_003')res = 0.2
if(sample_name=='rat_DA_01') res = 0.6
if(sample_name=='rat_Lew_01') res = 0.8
res_name = paste0('SCT_snn_res.', res)

seur <- FindNeighbors(seur, dims = 1:PC_NUMBER)
seur <- FindClusters(seur, resolution = res)
#### add the tSNE or UMAP components
sc = setDR(sc, getEmb(seur, 'tsne'))
sc = setClusters(sc, seur$SCT_snn_res.0.8)

#### checking the albumin expression 
plotMarkerMap(sc, 'Alb')

### manually setting the contamination fraction
#sc = setContaminationFraction(sc,0.2)
t_cell_markers = t_cell_markers[t_cell_markers%in% rownames(sc$toc)]
hep_markes = hep_markes[hep_markes %in% rownames(sc$toc)]
nonExpressedGeneList = list(Hepatocytes = hep_markes, T_cells = t_cell_markers)
plotMarkerDistribution(sc,nonExpressedGeneList)
plotMarkerDistribution(sc)

useToEst = estimateNonExpressingCells(sc, 
                                      nonExpressedGeneList = list(Hepatocytes = hep_markes), 
                                      clusters = seur$SCT_snn_res.0.8) #
### estimating the cells that should not express these genes and need to be used for the estimation
plotMarkerMap(sc, geneSet = hep_markes, useToEst = useToEst)
sc = calculateContaminationFraction(sc, list(Hepatocytes = hep_markes), useToEst = useToEst,forceAccept=TRUE)

out = adjustCounts(sc)
saveRDS(out, output_name)

seur_soupX = CreateSeuratObject(out)
seur_soupX = SCTransform(seur_soupX)

genes = c('Alb', 'Tat', 'G6pc', 'Cps1', 'Tdo2')
genes = c('Lyve1', 'Id3') # Lsecs
genes = c('Igfbp7', 'Rbp1', 'Col3a1', 'Sparc') # stellate cells
genes = c('Nkg7', 'Cd7')

gene_symbol = genes[2]
tsne_df = data.frame(Embeddings(seur, 'tsne'), gene_expr=GetAssayData(seur_soupX)[gene_symbol,])
ggplot(tsne_df, aes(tSNE_1, tSNE_2, color=gene_expr))+geom_point()+
  scale_color_viridis(direction = -1,option = "plasma")+theme_classic()+ggtitle(gene_symbol)

seur_soupX <- RunPCA(seur_soupX,verbose=T)
plot(100 * seur_soupX@reductions$pca@stdev^2 / seur_soupX@reductions$pca@misc$total.variance,
     pch=20,xlab="Principal Component",ylab="% variance explained",log="y")

seur_soupX <- RunTSNE(seur_soupX,dims=1:PC_NUMBER,reduction="pca",perplexity=30)
tsne_df = data.frame(Embeddings(seur_soupX, 'tsne'),clusters= seur$SCT_snn_res.0.6) 
ggplot(tsne_df, aes(tSNE_1, tSNE_2, color=clusters))+geom_point()+theme_classic()

seur_soupX <- RunUMAP(seur_soupX,dims=1:PC_NUMBER, reduction="pca")
umap_df = data.frame(Embeddings(seur_soupX, 'umap'),clusters= seur$SCT_snn_res.0.6) 
ggplot(umap_df, aes(UMAP_1, UMAP_2, color=clusters))+geom_point()+theme_classic()

Idents(seur_soupX) = seur$SCT_snn_res.0.6
clusters <- names(table(seur$SCT_snn_res.0.6))
Cluster_markers <- sapply(1:length(clusters), 
                          function(i) FindMarkers(seur_soupX, ident.1=clusters[i]), 
                          simplify = FALSE)
lapply(Cluster_markers, head)


#######################################
####### Automatic contamination fraction

## estimating the contamination fraction
#  contamination fraction estimate is the fraction of your data that will be discarded
sc = autoEstCont(sc)
# saveRDS(sc, 'soupX_estimcatedCont_10Xfilter.rds') ### this is using emptyDrop
clusters = sc$metaData$clusters
names(clusters) = colnames(sc$toc)
out = adjustCounts(sc, clusters=clusters) # method ='soupOnly'
getHead(out)
saveRDS(out, 'output_soupX_estimatedCont_10Xfilter.rds')
out <- readRDS('output_soupX_estimatedCont_10Xfilter.rds')

#####################################################################
### importing data
dir = 'Results/preproc_rats/'
Samples <- lapply(list.files(dir,pattern = '*_decontam.rds',full.names = T), readRDS)
names(Samples) <- c('rat_DA_M_10WK_003', 'rat_DA_01', 'rat_Lew_01', 'rat_Lew_02') #'rat_DA_02'
Samples <- lapply(Samples, CreateSeuratObject)
lapply(Samples, function(x) dim(x@assays$RNA))
num_cells <- lapply(Samples, function(x) ncol(x@assays$RNA))
genes <- unlist(lapply(Samples, function(x) rownames(x@assays$RNA)))

cluster_cell_type_df <- get_manual_labels()


old_Samples <- lapply(list.files(dir,pattern = '*00.rds',full.names = T), readRDS)
names(old_Samples) <- c('rat_DA_M_10WK_003', 'rat_DA_01', 'rat_Lew_01', 'rat_Lew_02') #'rat_DA_02'

### adding the final clusters to metadata
for(i in 1:length(old_Samples)){
  sample_name = names(old_Samples)[i]
  sample = old_Samples[[i]]
  
  if(sample_name=='rat_DA_02') 
    Idents(Samples[[i]]) <- paste0('cluster_',as.character(sample$SCT_snn_res.1))
  
  if(sample_name=='rat_Lew_01') 
    Idents(Samples[[i]]) <- paste0('cluster_',as.character(sample$SCT_snn_res.0.8))
  
  if(sample_name %in% c('rat_DA_01', 'rat_Lew_02', 'rat_DA_M_10WK_003')) 
    Idents(Samples[[i]]) <- paste0('cluster_',Idents(sample))
  
  Samples[[i]]$final_cluster <-Idents(Samples[[i]])
}

### adding predicted cell-type to meta-data
for( i in 1:length(Samples)){
  sample_name = names(Samples)[i]
  a_sample = Samples[[sample_name]]
  cluster_list = data.frame(umi=colnames(a_sample), cluster=a_sample$final_cluster, index=1:ncol(a_sample))
  map_df = cluster_cell_type_df[[sample_name]]
  merged = merge(cluster_list, map_df, by.x='cluster', by.y='cluster', all.x=T, sort=F)
  merged <- merged[order(merged$index, decreasing = F),]
  Samples[[sample_name]]$cell_type <- merged$cell_type
}

Samples <- lapply(Samples,  SCTransform)
### merging the samples
merged_samples <- merge(Samples[[1]], y = c(Samples[[2]], Samples[[3]], Samples[[4]]), 
                        add.cell.ids = names(Samples), project = "rat_data", 
                        merge.data = TRUE)

saveRDS(merged_samples_2, 'Results/preproc_rats/merged_samples_decontam.rds')
### check-ups on merge
sum(data.frame(num_cells)) == ncol(merged_samples@assays$SCT)
length(genes[!duplicated(genes)]) == nrow(merged_samples@assays$RNA)
DefaultAssay(merged_samples) <- 'RNA'

### finding variables genes and scaling the data 
# merged_samples <- FindVariableFeatures(merged_samples)
merged_samples <- ScaleData(merged_samples)
sample_type <- unlist(lapply(str_split(colnames(merged_samples),'_'), 
                             function(x) paste(x[-length(x)],collapse = '_')))
merged_samples$sample_type = sample_type
strain_names_df = data.frame(str_split_fixed(merged_samples$sample_type,pattern ='_',3))
merged_samples$strain_type = paste0(strain_names_df$X1,'_',strain_names_df$X2) 


## PCA
merged_samples <- FindVariableFeatures(merged_samples)
merged_samples <- RunPCA(merged_samples,verbose=T) #features=rownames(merged_samples)
plot(100 * merged_samples@reductions$pca@stdev^2 / merged_samples@reductions$pca@misc$total.variance,
     pch=20,xlab="Principal Component",ylab="% variance explained",log="y")
PC_NUMBER = 18
merged_samples <- RunHarmony(merged_samples, "sample_type",assay.use="RNA")

## UMAP
# running UMAP seperately: uwot, umapr, M3C
merged_samples <- RunUMAP(merged_samples,dims=1:PC_NUMBER, reduction="pca",perplexity=30,  reduction.name='umap')
merged_samples <- RunUMAP(merged_samples,dims=1:PC_NUMBER, reduction = "harmony",perplexity=30, reduction.name='umap_h')

umap_emb <- data.frame(Embeddings(merged_samples, 'umap_h'))
umap_emb$sample_type = merged_samples$sample_type
umap_emb$cell_type = merged_samples$cell_type
colnames(umap_emb)[1:2] <- c('UMAP_1', 'UMAP_2')

gene_name = 'Alb'
umap_emb$gene_exp = GetAssayData(merged_samples)[gene_name, ]


ggplot(umap_emb, aes(x=UMAP_1, y=UMAP_2))+
  geom_point(aes(color=sample_type),alpha=0.7,size=2)+theme_classic()
ggplot(umap_emb, aes(x=UMAP_1, y=UMAP_2))+
  geom_point(aes(color=cell_type),alpha=0.7,size=2)+theme_classic()
ggplot(umap_emb, aes(x=UMAP_1, y=UMAP_2))+
  geom_point(aes(color=gene_exp),alpha=0.7,size=2)+
  scale_color_viridis(direction = -1,option = "plasma")+theme_classic()+ggtitle(gene_name)

ggplot(umap_emb, aes(x=UMAP_1, y=UMAP_2))+
  geom_point(aes(shape=sample_type, color=cell_type),alpha=0.7,size=2)+
  theme_classic()+scale_colour_brewer(palette = "Paired")



###### strain differences analysis
cell_type_names = names(table(merged_samples$cell_type))
merged_samples_split <- sapply(1:length(cell_type_names), 
                               function(i){
                                 print(paste0(i, cell_type_names[i]))
                                 cell_type_indices = merged_samples$cell_type == cell_type_names[i]
                                 subset = CreateSeuratObject(merged_samples_data[,cell_type_indices])
                                 subset$sample_type = merged_samples$sample_type[cell_type_indices]
                                 subset$cell_type = merged_samples$cell_type[cell_type_indices]
                                 subset$strain = merged_samples$strain[cell_type_indices]
                                 return(subset)
                               }, 
                               simplify = F)
names(merged_samples_split) = cell_type_names

strain_markers_cellTypeSplit = sapply(1:length(merged_samples_split), function(i){
  print(names(merged_samples_split)[i])
  markers_df = FindMarkers(merged_samples_split[[i]], 
                           ident.1='rat_Lew', 
                           ident.2='rat_DA' , 
                           group.by='strain')
  markers_df$ensemble_ids = rownames(markers_df)
  ## merge the ensemble IDs in the dataframe with the HUGO terms 
  markers_df_merged <- merge(markers_df, 
                             mapper, 
                             by.x='ensemble_ids', 
                             by.y='V1', all.x=T, all.y=F,sort=F)
  return(markers_df_merged)
} , 
simplify = FALSE)

