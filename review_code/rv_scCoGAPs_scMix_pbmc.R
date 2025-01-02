
# cut, minNS,and maxNS control the process of matching patterns across subsets and should not be changed from defaults. 
# nSets should be less than or equal to the number of nodes/cores that are available. 
# some robustness can be lost when the subsets get too small. The general rule of thumb 
# is to set nSets so that each subset has between 1000 and 5000 genes or cells.
# Once the distributed parameters have been set we can call CoGAPS either by setting the distributed 
# parameter or by using the provided wrapper functions. The following calls are equivalent:

# ChiSq value tells us how closely the A and P matrices reconstruct the original data
### Qs:
# how many patterns need to be included and how to estimate its value?

#### running NIFA on the immune population of the new samples  #####
source('~/RatLiver/Codes/Functions.R')
#Initialize()
library(BiocParallel)
library(devtools)
library(CoGAPS)
library(NIFA)
library(fastICA)
#library(NMF)
library(Seurat)

scMix_3cl_merged = readRDS('~/scLMM/LMM-scRNAseq/Data/scMix_3cl_merged_logcounts.rds')
merged_samples <- scMix_3cl_merged

#################### scCoGAPS ####################
params <- new("CogapsParams")
#params <- setParam(params, "nIterations", 100)
num_patterns = 30
params <- setParam(params, "nPatterns", num_patterns) # set the value for a specific parameter
params <- setDistributedParams(params, nSets=detectCores()-2)
getParam(params, "nPatterns")

########### generating the input data ########### 
data <- as.matrix(GetAssayData(merged_samples))
# CoGAPS requires data to be genes x samples
Results.sc <- CoGAPS(data, params, distributed="single-cell", messages=T, transposeData=FALSE) #nIterations=1000
saveRDS(Results.sc, '~/sciRED/review_analysis/scCoGAPS_result_scMix.rds')
rm(data)
rm(Results.sc)



#Kang18_8vs8_seur = readRDS("~/scLMM/LMM-scRNAseq//Data/PBMC_Lupus_Kang8vs8_data_counts.rds")
#Kang18_8vs8_seur <- Seurat::SCTransform(Kang18_8vs8_seur, return.only.var.genes = FALSE, 
#                                        conserve.memory = FALSE, 
#                                        variable.features.n = nrow(Kang18_8vs8_seur),assay = 'originalexp')
Kang18_8vs8_seur = readRDS("~/scLMM/LMM-scRNAseq//Data/PBMC_Lupus_Kang8vs8_data_norm.rds")
merged_samples <- Kang18_8vs8_seur

params <- new("CogapsParams")
#params <- setParam(params, "nIterations", 100)
num_patterns = 30
params <- setParam(params, "nPatterns", num_patterns) # set the value for a specific parameter
params <- setDistributedParams(params, nSets=detectCores()-2)
getParam(params, "nPatterns")
########### generating the input data ########### 
data <- as.matrix(GetAssayData(merged_samples))
# CoGAPS requires data to be genes x samples
Results.sc <- CoGAPS(data, params, distributed="single-cell", messages=T, transposeData=FALSE) #nIterations=1000
saveRDS(Results.sc, '~/sciRED/review_analysis/scCoGAPS_result_pbmc.rds')










##################
head(Results.sc@featureLoadings)
head(Results.sc@sampleFactors)

################## making the umap  

scCoGaps <- data.frame(Results.sc@sampleFactors)
scCoGaps$umi <- rownames(scCoGaps)
colnames(merged_samples) == colnames(Results.sc@sampleFactors)
umap_df <- data.frame(Embeddings(merged_samples, 'umap'))
umap_df$umi <- rownames(umap_df)
umap_df <- merge(umap_df, scCoGaps, 'umi', 'umi', all.x=T)
umap_df$strain = unlist(lapply(str_split(umap_df$umi,pattern = '_' ), function(x) x[[2]]))
head(umap_df)

ggplot(umap_df, aes(UMAP_1,UMAP_2, color=Pattern_4 ))+geom_point()+theme_classic()+
  ggtitle('Pattern 4')+scale_color_viridis(direction = -1)
ggplot(umap_df[!is.na(umap_df$Pattern_1),], aes(x=strain, y=Pattern_7, fill=strain))+geom_boxplot()+
  theme_bw()+ggtitle('pattern 7')

################## making the tSNE  

tsne_df <- data.frame(Embeddings(merged_samples, 'tsne'))
tsne_df$umi <- rownames(tsne_df)
tsne_df <- merge(tsne_df, scCoGaps_immune, 'umi', 'umi', all.x=T)
tsne_df$strain = unlist(lapply(str_split(tsne_df$umi,pattern = '_' ), function(x) x[[2]]))
ggplot(tsne_df, aes(tSNE_1,tSNE_2, color=Pattern_6 ))+geom_point()+theme_classic()+
  ggtitle('Pattern 6')+scale_color_viridis(direction = -1)


plot(Results.sc)



####### ordering the genes corresponding to each factor #####
pattern_gene_list <- sapply(1:ncol(Results.sc@featureLoadings), function(i){
  pattern_index <- i
  pattern_gene_df <- data.frame(gene=rownames(Results.sc@featureLoadings), 
                                loading=Results.sc@featureLoadings[,pattern_index])
  pattern_gene_df <- pattern_gene_df[order(pattern_gene_df$loading, decreasing = T),]
  return(pattern_gene_df)
}, simplify = F)

lapply(pattern_gene_list, head, 25)

#### check the gene loadings of patterns of interest ####
pattern_of_interest = 4
head(pattern_gene_list[[pattern_of_interest]], 25)
write.csv(pattern_gene_list[[pattern_of_interest]], 
          file = paste0('Results/new_samples/coGAPs_pattern_',pattern_of_interest,'.csv'), 
          quote = F, row.names = F, col.names = T)

pheatmap(cor(as.data.frame(Results.sc@featureLoadings)))


pdf('Results/old_samples/scCoGAPS_varimax_cor.pdf')
############  correlation with varimax ############

### importing the varimax results 
rot_data <- readRDS('~/XSpecies/Results/preproc_rats/merged/rotated_Rat_PCs.rds')

#### correlation based on gene loadings ####

scCoGAPs.df <- data.frame(Results.sc@featureLoadings)
scCoGAPs.df$genes <- rownames(scCoGAPs.df)
head(scCoGAPs.df)

varimax_df <- data.frame(genes=rownames(rot_data$rotLoadings), 
           var_pc_5=rot_data$rotLoadings[,5], 
           var_pc_15=rot_data$rotLoadings[,15])

varimax_df <- merge(varimax_df, mapper, by.x='genes', by.y='V1', all.x=T)
merged_patterns <- merge(scCoGAPs.df, varimax_df, by.x='genes', by.y='V2')
cols_2_include <- colnames(merged_patterns)[c(2:8, 10, 11)]

merged_patterns_sub <- as.data.frame(merged_patterns[,cols_2_include])
pheatmap(cor(merged_patterns_sub, method = 'spearman'), main='Correlation based on gene loadings')
merged_patterns.m <- reshape2::melt(merged_patterns_sub)
ggplot(merged_patterns.m, aes(x=variable, y=value, fill=variable))+geom_boxplot()+theme_classic()
ggplot(merged_patterns_sub, aes(x=var_pc_5, y=Pattern_1))+geom_point()+ggtitle('gene loading correlation')
ggplot(merged_patterns_sub, aes(x=var_pc_5, y=Pattern_7))+geom_point()+ggtitle('gene loading correlation')
ggplot(merged_patterns_sub, aes(x=Pattern_1, y=Pattern_7))+geom_point()+ggtitle('gene loading correlation')


#### correlation based on embedding scores ####

scCoGAPs.df <- data.frame(Results.sc@sampleFactors)
scCoGAPs.df$umi <- rownames(scCoGAPs.df)
head(scCoGAPs.df)

varimax_df <- data.frame(umi=paste0(rownames(rot_data$rotScores), '-1'), 
                         #var_pc_5=rot_data$rotScores[,5], 
                         rot_data$rotScores[,1:15]
                         #var_pc_15=rot_data$rotScores[,15]
                         )

sum(!rownames(Results.sc@sampleFactors) %in% paste0(rownames(varimax_df), '-1'))

merged_patterns <- merge(scCoGAPs.df, varimax_df, by.x='umi', by.y='umi')
merged_patterns$strain = unlist(lapply(merged_patterns$umi, function(x) str_split_fixed(x, '_',n=6)[2]))
head(merged_patterns)
cols_2_include <- colnames(merged_patterns)[2:(ncol(merged_patterns)-1)]
merged_patterns_sub <- as.data.frame(merged_patterns[,cols_2_include])
cor_mat <- cor(merged_patterns_sub)[paste0('Pattern_',c(1:7)),paste0('X', 1:15)]
colnames(cor_mat) <- paste0('Var.PC ', c(1:15))
pheatmap(cor_mat, main='Correlation based on cell embeddings')

ggplot(merged_patterns, aes(x=X5, y=Pattern_1, color=strain))+geom_point()+
  ggtitle('cell embeddings correlation')+xlab('Varimax PC-5')+ylab('scCoGAPs Pattern-1')+theme_classic()
ggplot(merged_patterns, aes(x=X5, y=Pattern_7,color=strain))+geom_point()+ggtitle('cell embeddings correlation')
ggplot(merged_patterns, aes(x=X15, y=Pattern_1,color=strain))+geom_point()+ggtitle('cell embeddings correlation')

ggplot(merged_patterns, aes(y=X5, x=strain, fill=strain))+geom_boxplot()+ylab('Varimax PC-5')+ggtitle('')+theme_classic()
ggplot(merged_patterns, aes(x=X5, fill=strain))+geom_density(alpha=0.8)+xlab('Varimax PC-5')+ggtitle('')+theme_classic()

ggplot(merged_patterns, aes(y=Pattern_1, x=strain, fill=strain))+geom_boxplot()+ylab('scCoGAPs Pattern-1')+ggtitle('')+theme_classic()
ggplot(merged_patterns, aes(x=Pattern_1, fill=strain))+geom_density(alpha=0.8)+ylab('scCoGAPs Pattern-1')+ggtitle('')+theme_classic()

dev.off()



##### what exactly are these??
# get the unmatched patterns from each subset
unmatchedPatterns <- getUnmatchedPatterns(Results.sc)
length(unmatchedPatterns)
head(unmatchedPatterns[[1]])

# get the clustered patterns from the set of all patterns
clusteredPatterns <- getClusteredPatterns(Results.sc)
clusteredPatterns[[1]]

# get the correlation of each pattern to the cluster mean
cor_2_meanPattern <- getCorrelationToMeanPattern(Results.sc)

# get the size of the subsets used
sapply(getSubsets(Results.sc), length)

########################################################

K <- 8 ## number of latent factors
M <- 3 ## max number of mixture associated with each latent factor

NIFA.res <- apply(abs(cor(t(NIFAres$mu_S), t(simu$S))),2,max)
print(NIFA.res)

dim(X)
dim(tscale(X))
dim(NIFAres$S_expect)
dim(NIFAres$mu_S)
NIFAres <- NIFA(scale(X), # K = K, M = M,  
                S_threshold = 6e-5,
                init = "sd", A.init = NULL, S.init = NULL, verbose = T, 
                ref = t(simu$S), beta_expect_flag = NULL, L1.sd = NULL, 
                L2.sd = NULL, b_noise_prior = 1) # max.iter = 500,


merged_samples_sub <- readRDS('Results/new_samples/Immune_subclusters.rds') 
data <- as.matrix(GetAssayData(merged_samples_sub))

## using default values of the parameters for the 
NIFAres <- NIFA(data, # K = K, M = M,  
                S_threshold = 6e-5,
                init = "sd", A.init = NULL, S.init = NULL, verbose = T, 
                beta_expect_flag = NULL, L1.sd = NULL, 
                L2.sd = NULL, b_noise_prior = 1) # max.iter = 500,

NIFAres <- readRDS('Results/new_samples/NIFA_immuneCells.rds')
dim(data)
dim(NIFAres$S_expect)
dim(NIFAres$S_2_expect)
dim(NIFAres$mu_S) ## score matrix 
dim(NIFAres$mu_A) ## loading matrix
dim(NIFAres$mu_S_expect)
dim(NIFAres$mu_S_2_expect)
