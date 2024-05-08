library(reshape2)
library(ggplot2)

### import the results from the simulation (simulation_fc_multi.py) and visualize the results
# df = read.csv('~/sciFA/Results/simulation//metric_overlap_corr_df_sim100_April2024_v2.csv')
df = read.csv('~/sciRED/simulation/metric_overlap_corr_df_sim10.csv')
df = data.frame(t(df))
colnames(df) = df[1,]
df = df[-1,]
colnames(df)


df_melt = melt(t(df))
colnames(df_melt) = c('metric', 'overlap', 'R')
df_melt$metric = gsub('factor_','',df_melt$metric)
names(table(df_melt$metric))
df_melt$R = as.numeric(df_melt$R)

bimodality_metric = c('bimodality_index', 'calinski_harabasz','davies_bouldin', 'dip_score', "silhouette","wvrs" )
heterogeneity_metric=c("ASV_arith","ASV_geo") #"1-AUC_arith","1-AUC_geo",'ASV_simpson','ASV_entropy'
effect_size_metric=c('variance')
specificity_metric = c('entropy_fcat',  'simpson_fcat')

df_melt$metric_type[df_melt$metric %in% bimodality_metric]='Separability'
df_melt$metric_type[df_melt$metric %in% heterogeneity_metric]='Homogeneity'
df_melt$metric_type[df_melt$metric %in% effect_size_metric]='Effect size'
df_melt$metric_type[df_melt$metric %in% specificity_metric]='Specificity'
table(df_melt$metric_type)
table(df_melt$metric)

ggplot(df_melt, aes(x=metric,y=R))+geom_boxplot(notch = FALSE, fill='maroon')+
  coord_flip()+ylab('Correlation with overlap value')+
  theme(text = element_text(size=16),
        axis.text.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 15, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 15, angle = 90, hjust = .5, vjust = .5, face = "plain"))

table(df_melt$metric_type)
ggplot(df_melt, aes(x=metric,y=R,fill=metric_type))+
  geom_boxplot(notch = TRUE)+xlab('')+
  coord_flip()+ylab('Correlation with overlap value')+
  theme(text = element_text(size=16),
      axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
      axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
      axis.title.x = element_text(color = "grey20", size = 14, angle = 0, hjust = .5, vjust = 0, face = "plain"),
      axis.title.y = element_text(color = "grey20", size = 14, angle = 90, hjust = .5, vjust = .5, face = "plain"))

names(table(df_melt$metric_type))
df_melt_sub = df_melt[df_melt$metric_type== "Separability",] # "Homogeneity"  "Separability" "Specificity"
table(df_melt_sub$metric)
ggplot(df_melt_sub, aes(x=metric,y=R,fill=metric_type))+
  geom_boxplot(notch = TRUE)+xlab('')+scale_fill_manual(values = c('maroon'))+ #'cyan3' 'maroon' 'orange'
  coord_flip()+ylab('Correlation with overlap value')+ylim(-1, 1)+
  theme(text = element_text(size=16),
        axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 14, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 14, angle = 90, hjust = .5, vjust = .5, face = "plain"))

