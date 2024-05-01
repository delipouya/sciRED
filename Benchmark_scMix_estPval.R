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
fcat_single_list = lapply(list.files(file, pattern = "fcat_scMix*", full.names = T), read.csv)
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


cor_df = data.frame(imp=fcat_models_df_base$importance, model=fcat_pvalue_df_base$model)
cor_df_models<- split(cor_df, cor_df$model)
sapply(1:length(cor_df_models), function(i) colnames(cor_df_models[[i]])[1]<<-names(cor_df_models)[i])
cor_df_merged = Reduce(cbind, cor_df_models)
cor_df_merged <- cor_df_merged[,colnames(cor_df_merged) %in% names(cor_df_models)]
cor_mat = cor(cor_df_merged)
pheatmap::pheatmap(cor_mat, display_numbers = TRUE)
########### ########### ########### 

########### calculating empirical pvalues
fcat_pvalue_list = sapply(1:length(model_names), function(i){add_emp_pvalue(fcat_models_df, model_names[i])}, simplify = F)
names(fcat_pvalue_list) = model_names
ggplot(fcat_pvalue_list$DecisionTree, aes(x=pvalue))+geom_histogram(alpha=0.8, bins=100)+theme_classic()+ggtitle(a_model)

fcat_pvalue_df = Reduce(rbind, fcat_pvalue_list)
head(fcat_pvalue_df)

ggplot(fcat_pvalue_df, aes(x=model, y=pvalue, fill=model))+geom_boxplot(alpha=0.7)+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+coord_flip()

ggplot(fcat_pvalue_df, aes(x=pvalue, fill=model))+
  geom_density(alpha=0.5)+theme_classic()+scale_fill_brewer(palette = 'Set1')


sum(fcat_pvalue_df$pvalue[fcat_pvalue_df$model=='XGB'] < 0.05)
sum(fcat_pvalue_df$pvalue[fcat_pvalue_df$model=='RandomForest'] < 0.05)
sum(fcat_pvalue_df$pvalue[fcat_pvalue_df$model=='DecisionTree'] < 0.05)



###############################################################################################
########################## importance evaluation for model comparison
################################################################################################
meanimp_df_merged_deviance = read.csv('/home/delaram/sciFA/Results/residual/meanimp_df_scMixology_varimax_baseline_deviance.csv')
meanimp_df_residual_baseline$type = 'baseline'


file = '/home/delaram/sciFA/Results/residual/pearson/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T), read.csv)
meanimp_list_shuffle_pearson <- Reduce(rbind,meanimp_list)
meanimp_list_shuffle_pearson$res = 'pearson'
head(meanimp_list_shuffle_pearson)

file = '/home/delaram/sciFA/Results/residual/response/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T), read.csv)
meanimp_shuffle_response <- Reduce(rbind,meanimp_list)
meanimp_shuffle_response$res = 'response'
head(meanimp_shuffle_response)

file = '/home/delaram/sciFA/Results/residual/deviance//'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T), read.csv)
meanimp_shuffle_deviance <- Reduce(rbind,meanimp_list)
meanimp_shuffle_deviance$res = 'deviance'
head(meanimp_shuffle_deviance)

meanimp_residual_shuffle = rbind(rbind(meanimp_list_shuffle_pearson, meanimp_shuffle_response), meanimp_shuffle_deviance)
head(meanimp_residual_shuffle)
meanimp_residual_shuffle$type = 'shuffle'


meanimp_residual_merged = rbind(meanimp_df_residual_baseline, meanimp_residual_shuffle)
head(meanimp_residual_merged)
meanimp_residual_merged_m = melt(meanimp_residual_merged)
head(meanimp_residual_merged_m)


ggplot2::ggplot(meanimp_residual_merged_m, aes(y=value, x=res, fill=type))+geom_boxplot()+
  theme_classic()+coord_flip()+ylab('Mean importance value')





meanimp_df_residual_baseline_m = melt(meanimp_df_residual_baseline)
head(meanimp_df_residual_baseline_m)
meanimp_df_residual_shuffle_m = melt(meanimp_residual_shuffle)
head(meanimp_df_residual_shuffle_m)




mean_imp_baseline_m = data.frame(cov_level=meanimp_df_residual_baseline_m$X, 
                                 factor=meanimp_df_residual_baseline_m$variable,
                                 imp_score=meanimp_df_residual_baseline_m$value,
                                 res=meanimp_df_residual_baseline_m$res)
head(mean_imp_baseline_m)



importance_df_shuffle_split <- split(meanimp_df_residual_shuffle_m, meanimp_df_residual_shuffle_m$res)
lapply(importance_df_shuffle_split, head)
importance_df_baseline_split <- split(meanimp_df_residual_baseline_m, meanimp_df_residual_baseline_m$res)
lapply(importance_df_baseline_split, head)

names(importance_df_shuffle_split)
names(importance_df_baseline_split)

for(i in 1:length(importance_df_shuffle_split)){
  a_importance_df_shuffle = importance_df_shuffle_split[[i]]
  a_importance_df_basline = importance_df_baseline_split[[i]]
  importance_df_baseline_split[[i]]$pval = sapply(1:nrow(a_importance_df_basline), function(i) 
    sum(a_importance_df_shuffle$value>a_importance_df_basline$value[i])/nrow(a_importance_df_shuffle))
}

tab=rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.05))),
          pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.01))),
          pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.001))))

gridExtra::grid.table(t(tab))
dev.off()

tab=rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.05)/180,2))),
          pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.01)/180,2))),
          pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.001)/180,2))))
gridExtra::grid.table(t(tab))

thr = 0.01
sapply(1:length(importance_df_baseline_split), function(i) 
{importance_df_baseline_split[[i]]$sig <<- importance_df_baseline_split[[i]]$pval < thr})
head(importance_df_baseline_split[[1]])

AvgFacSig_df_model = sapply(1:length(importance_df_baseline_split), function(i){
  a_model_imp_df = importance_df_baseline_split[[i]]
  a_model_imp_df_cov = split(a_model_imp_df, a_model_imp_df$X)
  AvgFacSig = sapply(1:length(a_model_imp_df_cov), function(i){
    sum(a_model_imp_df_cov[[i]]$sig)
  })
  names(AvgFacSig) = names(a_model_imp_df_cov)
  return(AvgFacSig)
}, simplify = T)

colnames(AvgFacSig_df_model) = names(importance_df_baseline_split) 
head(AvgFacSig_df_model)
AvgFacSig_df_model_m = melt(AvgFacSig_df_model)
head(AvgFacSig_df_model_m)

ggplot(AvgFacSig_df_model_m, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))

