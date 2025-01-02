source('~/RatLiver/Codes/Functions.R')
Initialize()
library(slalom)
library(ggplot2)
theme_set(theme_bw(12))
library(slalom)


## currently running on the immune population of the new samples on the runners screen 

#### loading the input data ####
## loading the immune population of the new rat samples
#merged_samples_sub <- readRDS('~/RatLiver/Results/new_samples/merged_samples_immune_sub.rds')
merged_samples_sub <- readRDS('Results/new_samples/Immune_subclusters.rds') 


## loading the complete dataset of the old rat samples
# merged_samples <- readRDS('Objects/merged_samples_oldSamples.rds')

### subsetting the highly varible genes
merged_samples_sub <- FindVariableFeatures(merged_samples_sub, selection.method = "vst", nfeatures = 2000)

input_data <- merged_samples_sub[VariableFeatures(merged_samples_sub),]
dim(input_data)

### converting the input data to sce object
exprs_matrix <- as.matrix(GetAssayData(input_data)) ### accessing the normalized data
data <- SingleCellExperiment::SingleCellExperiment(
  assays = list(logcounts = exprs_matrix)
)
dim(data)


# gmtfile <- system.file("extdata", "reactome_subset.gmt", package = "slalom")
gmtfile <- '~/Rat_GOBP_AllPathways_no_GO_iea_May_01_2020_symbol.gmt'
genesets0 <- GSEABase::getGmt(gmtfile)
length(genesets0)
num_genesets = 4000
geneset_indices = unique(round(runif(n = num_genesets, min = 1, max = length(genesets0))))
genesets = genesets0[geneset_indices]

names(genesets)

## Slalom fits relatively complicated hierarchical Bayesian factor analysis models
## with data and results stored in a "SlalomModel" object. 
model <- newSlalomModel(data, genesets, n_hidden = 5, min_genes = 10)

## start measuring time
start.time <- Sys.time()

## Initialize a SlalomModel with sensible starting values for parameters before training the model.
model <- initSlalom(model)

## Train a SlalomModel to infer model parameters.
model <- trainSlalom(model, nIterations = 10000)#, nIterations = 1000

## stop measuring time
end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)


## adding the info back to the sce object
data <- addResultsToSingleCellExperiment(data, model, n_active=1e20, annotated=T,
                                         unannotated_dense=T, unannotated_sparse=T,mad_filter=0.4,
                                         add_loadings=T, dimred='slalom', check_convergence=F)
run_number = paste0(num_genesets, '_gs')
### saving the results
saveRDS(data, paste0('~/RatLiver/Results/new_samples/fsclvm_model_immuneCells_',run_number,'_2.rds'))



data <- readRDS('~/RatLiver/Results/new_samples/fsclvm_model_immuneCells_.rds')
 
### accessing embedding and loading matrix
loading.slalom <- data.frame(rowData(data))
embedding.slalom <- data.frame(reducedDim(data, 'slalom'))
dim(embedding.slalom)
embedding.slalom$cluster= merged_samples_sub$immune_clusters
embedding.slalom$sample= merged_samples_sub$sample_name


head(loading.slalom)
head(embedding.slalom)

dim(loading.slalom)
dim(embedding.slalom)

###### Evaluating the results ###### 
## data.frame with factors ordered by relevance, showing term
model.topTerms = topTerms(model, n_active=1e20, mad_filter=0.4, annotated=T, 
                          unannotated_dense=T, unannotated_sparse=T)

model.topTerms <- readRDS('~/RatLiver/Results/new_samples/fsclvm_model_immuneCells_topTerms.rds')
head(model.topTerms)

ggplot(embedding.slalom, aes(REGULATION.OF.LEUKOCYTE.PROLIFERATION.GOBP.GO.0070663, 
                             HALLMARK_ALLOGRAFT_REJECTION.MSIGDB_C2.HALLMARK_ALLOGRAFT_REJECTION, color=cluster))+geom_point()+theme_classic()

ggplot(embedding.slalom, aes(REGULATION.OF.LEUKOCYTE.PROLIFERATION.GOBP.GO.0070663, 
                             HALLMARK_ALLOGRAFT_REJECTION.MSIGDB_C2.HALLMARK_ALLOGRAFT_REJECTION, color=sample))+geom_point()+theme_classic()

model.topTerms.df <- data.frame(model.topTerms)
head(model.topTerms.df)


pdf('~/RatLiver/Plots/fsclvm_newSamples_immuneCells.pdf',width = 15, height = 15)
for(i in 1:(ncol(embedding.slalom)-2)){
  
  p1=ggplot(embedding.slalom, aes(y=embedding.slalom[,i], x=sample,fill=sample))+geom_violin()+theme_classic()+ylab(colnames(embedding.slalom)[i])
  p2=ggplot(embedding.slalom, aes(embedding.slalom[,i],fill=sample))+geom_density(alpha=0.6)+theme_classic()+xlab(colnames(embedding.slalom)[i])
  p3=ggplot(embedding.slalom, aes(y=embedding.slalom[,i], x=embedding.slalom[,1],color=sample))+
    geom_point()+theme_classic()+ylab(colnames(embedding.slalom)[i])+xlab(colnames(embedding.slalom)[1])
  p4=ggplot(embedding.slalom, aes(y=embedding.slalom[,i], x=embedding.slalom[,1],color=cluster))+
    geom_point()+theme_classic()+ylab(colnames(embedding.slalom)[i])+xlab(colnames(embedding.slalom)[1])
  
  gridExtra::grid.arrange(p1,p2,p3,p4,nrow=2,ncol=2)
  
}
dev.off()


#### visualizing the results - saving the model itself is not possible
## Plot relevance for all terms
plotRelevance(model)
## Plot highest loadings of a factor
plotLoadings(model, "CELL_CYCLE") 

########### evaluating the run-time
library(readxl)
fscLVM_runtime <- data.frame(read_excel("~/fscLVM_runtime.xlsx"))
fscLVM_runtime = fscLVM_runtime[-nrow(fscLVM_runtime),]
fit <- lm(run_time_h ~ retained_gs, data = fscLVM_runtime)
fit_info = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                 "Intercept =",signif(fit$coef[[1]],5 ),
                 " Slope =",signif(fit$coef[[2]], 5),
                 " P =",signif(summary(fit)$coef[2,4], 5))
head(fscLVM_runtime)      
ggplot(fscLVM_runtime, aes(x=retained_gs, y=run_time_h, color=final_gs))+
  geom_point(size=3)+scale_color_continuous(name='# Inferred\nFactors')+
  xlab('Number of retained genesets\n(after dropping uninformative input genesets)')+ylab('Run time (h)')+
  ggtitle(paste0('Run time of fscLVM with increasing number of input genesets\ninput data dimentions: 2000 genes - 978 cells'),fit_info)+
  geom_smooth(method='lm',  color='turquoise4', size=0.4, se=F)

#### predicting the run-time for a given number of genesets
new = data.frame(retained_gs=c(20000))
predict(fit,new )



fit <- lm(run_time_h ~ intial_gs, data = fscLVM_runtime)
fit_info = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                 "Intercept =",signif(fit$coef[[1]],5 ),
                 " Slope =",signif(fit$coef[[2]], 5),
                 " P =",signif(summary(fit)$coef[2,4], 5))
head(fscLVM_runtime)      
ggplot(fscLVM_runtime, aes(x=intial_gs, y=run_time_h, color=final_gs))+
  geom_point(size=3)+scale_color_continuous(name='# Inferred\nFactors')+
  xlab('Number of raw genesets\n(Before dropping uninformative input genesets)')+ylab('Run time (h)')+
  ggtitle(paste0('Run time of fscLVM with increasing number of input genesets\ninput data dimentions: 2000 genes - 978 cells'),fit_info)+
  geom_smooth(method='lm',  color='red', size=0.4, se=F)

new = data.frame(intial_gs=c(20000))
predict(fit,new )


fit <- lm(retained_gs ~ intial_gs, data = fscLVM_runtime)
new = data.frame(intial_gs=c(20000))
predict(fit,new )


