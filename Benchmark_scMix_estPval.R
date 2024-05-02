library(ggplot2)
library(ggpubr)
library(reshape2)

scale_minMax <- function(x){
  x_min = min(x)
  x_max = max(x)
  scaled = (x-x_min)/(x_max-x_min)
  return(scaled)
}

scale_Max <- function(x){
  x_max = max(x)
  scaled = (x)/(x_max)
  return(scaled)
}


add_emp_pvalue <- function(fcat_df, a_model){
  ### input: dataframe of merged fcat scores for shuffled and baseline fca scores. 
  ### counts the number of observations in the empirical null distribution which are higher than the given fca score (fca_emp_h)
  ### calculates the empirical p-value by dividing the fca_emp_h by the total number of null dist observarions
  if(!'importance' %in% colnames(fcat_df)){
    colnames(fcat_df)[colnames(fcat_df)=="value"]='importance'
  }
  
  fcat_df_shuffle= fcat_df[fcat_df$type == 'shuffle',]
  null_empirical_dist = fcat_df_shuffle$importance[fcat_df_shuffle$model==a_model]
  
  model_fcat_base = fcat_df[fcat_df$type == 'baseline' & fcat_df$model == a_model,]
  model_fcat_base$pvalue = sapply(1:nrow(model_fcat_base), 
                                  function(i) sum(null_empirical_dist>model_fcat_base$importance[i])/length(null_empirical_dist), 
                                  simplify = T)
  return(model_fcat_base)
}



fcat_single_base = read.csv('/home/delaram/sciRED/benchmark/scMix/baseline/fcat_scMix_single_baseline.csv')
fcat_single_base$type = 'baseline'

file = '/home/delaram/sciRED/benchmark/scMix/shuffle/single/'
fcat_single_list = lapply(list.files(file, pattern = "fcat_scMix_single*", full.names = T), read.csv)
fcat_single_shuffle <- Reduce(rbind,fcat_single_list)
fcat_single_shuffle$type = 'shuffle'
head(fcat_single_shuffle)

fcat_single = rbind(fcat_single_base, fcat_single_shuffle)
fcat_single$importance_abs = abs(fcat_single$importance)

ggplot(fcat_single, aes(x=model, y=importance, fill=type))+
  geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#999999", "maroon"))

fcat_models<- split(fcat_single, fcat_single$model)
#### scaling various classifier scores
sapply(1:length(fcat_models), function(i) {fcat_models[[i]]$imp_scale <<- scale(fcat_models[[i]]$importance, center = FALSE)}, simplify = F)
sapply(1:length(fcat_models), function(i) {fcat_models[[i]]$imp_z_trans <<- scale(fcat_models[[i]]$importance)}, simplify = F)
sapply(1:length(fcat_models), function(i) {fcat_models[[i]]$imp_minmax <<- scale_minMax(fcat_models[[i]]$importance)}, simplify = F)
sapply(1:length(fcat_models), function(i) {fcat_models[[i]]$imp_max_scale <<- scale_Max(fcat_models[[i]]$importance)}, simplify = F)


###### Figure B for the benchmark panel
fcat_models_df = Reduce(rbind, fcat_models)
ggplot(fcat_models_df, aes(x=model, y=importance, fill=type))+geom_boxplot()+
  theme_classic()+coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+
  theme(text = element_text(size=18))+xlab('')

ggplot(fcat_models_df, aes(x=model, y=imp_minmax, fill=type))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+
  theme(text = element_text(size=18))+xlab('')+ylab('Importance score (min-max scaled)')


########### sanity check ########### 
fcat_models_df_base= fcat_models_df[fcat_models_df$type == 'baseline',]
fcat_models_df_shuffle = fcat_models_df[fcat_models_df$type == 'shuffle',]

model_names = names(table(fcat_models_df_shuffle$model))
ggplot(fcat_models_df_shuffle, aes(x=importance, fill=model))+
  geom_histogram(alpha=0.5,color='black',bins=100)+theme_classic()+scale_fill_brewer(palette = 'Set1')

ggplot(fcat_models_df_shuffle, aes(x=imp_minmax, fill=model))+
  geom_histogram(alpha=0.5,color='black',bins=100)+theme_classic()+scale_fill_brewer(palette = 'Set1')

a_model = "RandomForest"
model_imp_shuffle_values = fcat_models_df_shuffle$importance[fcat_models_df_shuffle$model==a_model]
ggplot(fcat_models_df_shuffle, aes(x=importance))+geom_histogram( bins=200,fill='grey')+
  theme_classic()+ggtitle(a_model)+theme(text = element_text(size=18))+xlab('FCA scores for a single model')+
  ylab("Frequency")+geom_vline(xintercept=0.09, color = "red", size=1, linetype="dashed")


cor_df = data.frame(imp=fcat_models_df_base$importance, model=fcat_models_df_base$model)
cor_df_models<- split(cor_df, cor_df$model)
sapply(1:length(cor_df_models), function(i) colnames(cor_df_models[[i]])[1]<<-names(cor_df_models)[i])
cor_df_merged = Reduce(cbind, cor_df_models)
cor_df_merged <- cor_df_merged[,colnames(cor_df_merged) %in% names(cor_df_models)]
cor_mat = cor(cor_df_merged)
pheatmap::pheatmap(cor_mat, display_numbers = TRUE)
########### ########### ########### 

########### calculating empirical p-values
fcat_pvalue_list = sapply(1:length(model_names), function(i){add_emp_pvalue(fcat_models_df, model_names[i])}, simplify = F)
names(fcat_pvalue_list) = model_names
fcat_pvalue_df = Reduce(rbind, fcat_pvalue_list)

model_type_names = c("AUC","DecisionTree","LogisticRegression","XGB")
model_type_names = names(table(fcat_pvalue_df$model))
thr = 0.05
cov_level_names = names(table(fcat_pvalue_df$covariate_level))
summary_df = data.frame(matrix(nrow =length(model_type_names),ncol = length(cov_level_names)))
colnames(summary_df) = cov_level_names
summary_df

row=1
thr = 0.05
for (model_type in model_type_names){
  basline_df  = fcat_pvalue_df[fcat_pvalue_df$model == model_type,]
  #### defining which elements in baseline are considered as significant based on the threshold
  sapply(1:length(basline_df), function(i) {basline_df$sig <<- basline_df$pval < thr})
  
  a_model_imp_df_cov = split(basline_df, basline_df$covariate_level)
  AvgFacSig_per_cov = sapply(1:length(a_model_imp_df_cov), function(i){
    sum(a_model_imp_df_cov[[i]]$sig)
  })
  names(AvgFacSig_per_cov) = names(a_model_imp_df_cov)
  summary_df[row,]=AvgFacSig_per_cov
  rownames(summary_df)[row]=model_type
  row=row+1
}
summary_df


single_fcat_sum = melt( t(summary_df))
head(single_fcat_sum)
##### checking if it worked for single classifiers
ggplot(single_fcat_sum, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))



###############################################################################################
########################## importance evaluation for model comparison
################################################################################################

fcat_mean_base = read.csv('/home/delaram/sciRED/benchmark/scMix/baseline/fcat_scMix_mean_baseline.csv')
fcat_mean_base$type = 'baseline'
head(fcat_mean_base)
test = melt(fcat_mean_base)

file = '/home/delaram/sciRED/benchmark/scMix/shuffle/mean/'
fcat_mean_list = lapply(list.files(file, pattern = "fcat_scMix_mean*", full.names = T), read.csv)
fcat_mean_shuffle <- Reduce(rbind,fcat_mean_list)
fcat_mean_shuffle$type = 'shuffle'
head(fcat_mean_shuffle)

fcat_mean = rbind(fcat_mean_base, fcat_mean_shuffle)
fcat_mean_m = melt(fcat_mean)
ggplot(fcat_mean_m, aes(y=value, x=type, fill=type))+geom_boxplot()+coord_flip()+ylab('Mean fcat')



fcat_mean_m$model = paste0(fcat_mean_m$scale_type, '_', fcat_mean_m$mean_type)
model_names = names(table(fcat_mean_m$model))
########### calculating empirical p-values
meanfcat_pvalue_list = sapply(1:length(model_names), function(i){add_emp_pvalue(fcat_mean_m, model_names[i])}, simplify = F)
names(meanfcat_pvalue_list) = model_names
meanfcat_pvalue_pvalue_df = Reduce(rbind, meanfcat_pvalue_list)

thr = 0.05
colnames(meanfcat_pvalue_pvalue_df)[colnames(meanfcat_pvalue_pvalue_df)=='X']='covariate_level'

cov_level_names = names(table(meanfcat_pvalue_pvalue_df$covariate_level))
summary_df = data.frame(matrix(nrow =length(model_names),ncol = length(cov_level_names)))
colnames(summary_df) = cov_level_names

row=1
for (model_type in model_names){
  basline_df  = meanfcat_pvalue_pvalue_df[meanfcat_pvalue_pvalue_df$model == model_type,]
  #### defining which elements in baseline are considered as significant based on the threshold
  sapply(1:length(basline_df), function(i) {basline_df$sig <<- basline_df$pval < thr})
  
  a_model_imp_df_cov = split(basline_df, basline_df$covariate_level)
  AvgFacSig_per_cov = sapply(1:length(a_model_imp_df_cov), function(i){
    sum(a_model_imp_df_cov[[i]]$sig)
  })
  names(AvgFacSig_per_cov) = names(a_model_imp_df_cov)
  summary_df[row,]=AvgFacSig_per_cov
  rownames(summary_df)[row]=model_type
  row=row+1
}
summary_df

mean_fcat_sum = melt( t(summary_df))
head(mean_fcat_sum)
mean_fcat_sum$Var2 = as.character(mean_fcat_sum$Var2)
##### checking if it worked for single classifiers
ggplot(mean_fcat_sum, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))


###################################

single_fcat_sum$model_type =  'single'
mean_fcat_sum$model_type =  'ensemble'
head(single_fcat_sum)
head(mean_fcat_sum)

both_fcat_sum = rbind(single_fcat_sum, mean_fcat_sum)
both_fcat_sum$Var2 = as.character(both_fcat_sum$Var2)
both_fcat_sum$Var2[both_fcat_sum$Var2=='standard_arithmatic']='sciRED'
head(both_fcat_sum)

both_fcat_sum_single = both_fcat_sum[both_fcat_sum$model_type=='single' | both_fcat_sum$Var2=='sciRED',]

both_fcat_sum_mean = both_fcat_sum[both_fcat_sum$model_type=='ensemble' | 
                                     both_fcat_sum$Var2=='sciRED',]


ggplot(both_fcat_sum_single, aes(y=value,x=reorder(Var2, value), fill=model_type))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17),
                     axis.text.y = element_text(size=19.5),axis.text.x = element_text(size=18)
  )+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  #geom_ribbon(aes(ymin = 0, ymax = 3), fill = "grey70") +
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  #geom_hline(yintercept=2, color = "red", size=1, linetype="dashed")+
  #geom_hline(yintercept=3, color = "red", size=1, linetype="dashed")+
  geom_area(mapping = aes(y = ifelse(value>0 & value< 3 , 1, 0)), fill = "grey70")
#ggtitle(paste0('pvalue threshold=',thr))
