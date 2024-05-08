library(reshape2)
library(data.table)


#######################################################
##### Human liver base model
#######################################################

gini_base_meanimp =list(
  'pearson'= list('arithmatic_minmax'= (0.4040987141830408),
                  'arithmatic_rank'= (0.20181721483005238),
                  'arithmatic_standard'= (0.2988997215338046),
                  'geometric_minmax'= (0.661369124472936),
                  'geometric_rank'= (0.24888654933526397),
                  'geometric_standard'= (0.32081240738701083)),
  
  'response'= list('arithmatic_minmax'= (0.4230967545767304),
                   'arithmatic_rank'= (0.20372265617270913),
                   'arithmatic_standard'= (0.3168410173954648),
                   'geometric_minmax'= (0.6829727215270417),
                   'geometric_rank'= (0.2490793883170096),
                   'geometric_standard'= (0.3335889875876788)),
  
  'deviance'= list('arithmatic_minmax'= (0.3994863019950993),
                   'arithmatic_rank'= (0.19917652172661915),
                   'arithmatic_standard'= (0.311990244199414),
                   'geometric_minmax'= (0.6953281982506246),
                   'geometric_rank'= (0.24264052176003112),
                   'geometric_standard'= (0.33752263791616005)))


gini_base_imp = list(
  'pearson'=list('AUC'=(0.40807551491587996),
                 'DecisionTree'= (0.7780708936634838),
                 'LogisticRegression'= (0.5864067891947421),
                 'XGB'= (0.5589863667590491)),
  
  'response'= list('AUC'= (0.41912129978784196),
                   'DecisionTree'= (0.793771858252523),
                   'LogisticRegression'= (0.6802298211456309),
                   'XGB'= (0.5667835615585536)),
  
  'deviance'= list('AUC'= (0.39170804243377344),
                   'DecisionTree'= (0.8107653361729017),
                   'LogisticRegression'= (0.4917192403776668),
                   'XGB'= (0.6151741640722693)))
#######################################################


#### visulize gini distributions for the human liver data
setwd('/home/delaram/sciFA/Results/benchmark_humanliver/gini_analysis/')

files_list_meanimp = list.files(pattern = 'meanimp_gini_*')
list_meanimp = lapply(files_list_meanimp, read.csv)
names(list_meanimp) = files_list_meanimp
lapply(list_meanimp, head)
list_meanimp_merged = Reduce(rbind, list_meanimp)
list_meanimp_merged = list_meanimp_merged[,-1]
head(list_meanimp_merged)
list_meanimp_merged_m = melt(list_meanimp_merged)
head(list_meanimp_merged_m)


files_list_imp = list.files(pattern = 'imp_gini_*')[1:3]
list_imp = lapply(files_list_imp, read.csv)
names(list_imp) = files_list_imp
lapply(list_imp, dim)
list_imp_merged = Reduce(rbind, list_imp)
list_imp_merged = list_imp_merged[,-1]
head(list_imp_merged)
list_imp_merged_m = melt(list_imp_merged)
head(list_imp_merged_m)

merged_all = rbind(list_imp_merged_m, list_meanimp_merged_m)

####################### formating the base datapints to be added to plot
gini_base_imp_df = data.frame(rbindlist(gini_base_imp, fill=TRUE))
gini_base_imp_df$residual_type = names(gini_base_imp)
gini_base_imp_df_melt = melt(gini_base_imp_df)
gini_base_imp_df_melt

gini_base_meanimp_df = data.frame(rbindlist(gini_base_meanimp, fill=TRUE))
gini_base_meanimp_df$residual_type = names(gini_base_meanimp)
gini_base_meanimp_df_melt = melt(gini_base_meanimp_df)
gini_base_meanimp_df_melt

merged_all_base = rbind(gini_base_imp_df_melt, gini_base_meanimp_df_melt)


ggplot(list_imp_merged_m, aes(x=reorder(variable, value), y=value, fill=residual_type))+geom_boxplot()+
  theme_classic()+coord_flip()+theme(text = element_text(size=17))+xlab('')+
  scale_fill_brewer(palette = 'Set1')+ylab('Gini index')+
  geom_point(data = gini_base_imp_df_melt, color = "goldenrod2", 
             position =  position_dodge(width = .75), size = 2)


merged_all$variable = as.character(merged_all$variable)
merged_all$variable[merged_all$variable=='arithmatic_standard'] = 'sciRED'

merged_all_base$variable = as.character(merged_all_base$variable)
table(merged_all$variable)
merged_all_base$variable[merged_all_base$variable=='arithmatic_standard']='sciRED'

merged_all = merged_all[merged_all$model %in% 'ensemble',]
merged_all_base[merged_all_base$variable %in% merged_all$variable,]
ggplot(merged_all, aes(x=reorder(variable, value), y=value, fill=residual_type))+geom_boxplot()+
  theme_classic()+theme(text = element_text(size=17))+xlab('')+coord_flip()+
  scale_fill_brewer(name='Residual Type',palette = 'Set1')+ylab('Gini index')+
  geom_point(data = merged_all_base, color = "red3", 
             position =  position_dodge(width = .75), size = 1.3)
#geom_point(aes(colour = Cell_line, shape = replicate, group = Cell_line),
#           position = position_dodge(width = .75), size = 3)





#######################################################
########### scMixology ############################################
#######################################################


gini_base_imp =list(
  'pearson'= list('AUC'= c(0.30747268687955204),
              'DecisionTree'= c(0.7671690241771978),
              'KNeighbors_permute'= c(0.43300978074699975),
              'LogisticRegression'=c (0.46593939027637116),
              'RandomForest'= c(0.4342762106674533),
              'XGB'= c(0.6059417173359738)))

gini_base_meanimp =list(
  'pearson'= list('arithmatic_minmax'= (0.29806550005016647),
                  'arithmatic_rank'= (0.14965514190435478),
                  'arithmatic_standard'= (0.21324298837017963),
                  'geometric_minmax'= (0.638038589924875),
                  'geometric_rank'= (0.1773109372125837),
                  'geometric_standard'= (0.23318375950541462)))

#######################################################


#### visulize gini distributions for the human liver data
setwd('/home/delaram/sciFA/Results/benchmark/gini_analysis/')

files_list_meanimp = list.files(pattern = 'meanimp_gini_*')
list_meanimp = lapply(files_list_meanimp, read.csv)
names(list_meanimp) = files_list_meanimp
lapply(list_meanimp, head)
list_meanimp_merged = Reduce(rbind, list_meanimp)
list_meanimp_merged = list_meanimp_merged[,-1]
head(list_meanimp_merged)
list_meanimp_merged_m = melt(list_meanimp_merged)
head(list_meanimp_merged_m)


files_list_imp = list.files(pattern = 'imp_gini_*')[1]
list_imp = lapply(files_list_imp, read.csv)
names(list_imp) = files_list_imp
lapply(list_imp, dim)
list_imp_merged = Reduce(rbind, list_imp)
list_imp_merged = list_imp_merged[,-1]
head(list_imp_merged)
list_imp_merged_m = melt(list_imp_merged)
head(list_imp_merged_m)

merged_all = rbind(list_imp_merged_m, list_meanimp_merged_m)
table(merged_all$model)
table(merged_all$variable)
merged_all$variable = as.character(merged_all$variable)
merged_all$variable[merged_all$variable=='arithmatic_standard'] = 'sciRED'

####################### formating the base datapints to be added to plot
gini_base_imp_df = data.frame(rbindlist(gini_base_imp, fill=TRUE))
gini_base_imp_df$residual_type = names(gini_base_imp)
gini_base_imp_df_melt = melt(gini_base_imp_df)
gini_base_imp_df_melt

gini_base_meanimp_df = data.frame(rbindlist(gini_base_meanimp, fill=TRUE))
gini_base_meanimp_df$residual_type = names(gini_base_meanimp)
gini_base_meanimp_df_melt = melt(gini_base_meanimp_df)
gini_base_meanimp_df_melt

merged_all_base = rbind(gini_base_imp_df_melt, gini_base_meanimp_df_melt)
merged_all_base$model = c(rep('single',6),rep('ensemble',6))
merged_all_base = merged_all_base[!merged_all_base$variable %in% c('KNeighbors_permute',  'RandomForest'),]

merged_all_base$variable = as.character(merged_all_base$variable)
table(merged_all$variable)
merged_all_base$variable[merged_all_base$variable=='arithmatic_standard']='sciRED'

ggplot(merged_all, aes(x=reorder(variable, value), y=value, fill=model))+geom_boxplot()+
  theme_classic()+theme(text = element_text(size=17))+xlab('')+coord_flip()+
  scale_fill_brewer(palette = 'Set1')+ylab('Gini index')+
  geom_point(data = merged_all_base, color = "red3", 
             position =  position_dodge(width = .75), size = 1.3)
#geom_point(aes(colour = Cell_line, shape = replicate, group = Cell_line),
#           position = position_dodge(width = .75), size = 3)


