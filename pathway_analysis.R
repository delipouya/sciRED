source('~/RatLiver/Codes/Functions.R')
Initialize()
library(gprofiler2)
library(gridExtra)
library(tidyverse)

get_gprofiler_enrich <- function(markers, model_animal_name){
  gostres <- gost(query = markers,
                  sources=c('GO:BP' ,'REAC'),
                  organism = model_animal_name,
                  ordered_query = FALSE, 
                  multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                  measure_underrepresentation = FALSE, evcodes = TRUE, 
                  user_threshold = 0.05, correction_method = "g_SCS", 
                  domain_scope = "annotated", custom_bg = NULL, 
                  numeric_ns = "",  highlight = TRUE)
  return(gostres)
}
upload_GMT_file(gmtfile = '~/c7.immunesigdb.v2023.2.Hs.symbols.gmt') #"gp__SEA8_T0ld_VHU"
model_animal_name ='hsapiens'

#################################################################################
########################    Human Kidney Map ###########################
####################################################################################
varimax_df = read.csv('~/sciFA/Results/varimax_loading_df_kidneyMap.csv')
df = data.frame(gene= varimax_df$X,factor=varimax_df$F18)
df_pos = df[order(df$factor, decreasing = T),]
num_genes = 200

table_to_vis = df_pos[1:20,]
rownames(table_to_vis) = NULL
colnames(table_to_vis) = c('Gene', 'Score')
table_to_vis$Score = round(table_to_vis$Score, 3)

dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(table_to_vis, theme=tt2)


enrich_res = get_gprofiler_enrich(markers=df_pos$gene[1:num_genes], model_animal_name)
enrich_res_pos = data.frame(enrich_res$result)
enrich_res_pos = enrich_res_pos[,!colnames(enrich_res_pos)%in%'evidence_codes']
enrich_res_pos$log_p = -log(as.numeric(enrich_res_pos$p_value))
enrich_res_pos = enrich_res_pos[order(enrich_res_pos$log_p, decreasing = T),]
View(data.frame(1:nrow(enrich_res_pos),enrich_res_pos$term_name, enrich_res_pos$intersection))

enrich_res_pos = enrich_res_pos[c(1,2,7,8,11,14,20),] ## positive - F18 - kidney dataset
enrich_res_pos = enrich_res_pos[,colnames(enrich_res_pos) %in% c('term_name', 'log_p')]

enrich_res_pos$term_name = gsub('metabolic process', 'metabolism',enrich_res_pos$term_name)
enrich_res_pos$term_name <- factor(enrich_res_pos$term_name, levels =  enrich_res_pos$term_name[length(enrich_res_pos$term_name):1])

title = ''#'stim'#'Male'
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity',fill='palevioletred',color='grey10')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  scale_fill_manual(values = c('palevioletred'))+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 15, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))


df_neg = df[order(df$factor, decreasing = F),]
enrich_res = get_gprofiler_enrich(markers=df_neg$gene[1:num_genes], model_animal_name)
enrich_res_neg = data.frame(enrich_res$result)
enrich_res_neg = enrich_res_neg[,!colnames(enrich_res_neg)%in%'evidence_codes']
enrich_res_neg$log_p = -log(as.numeric(enrich_res_neg$p_value))
enrich_res_neg = enrich_res_neg[order(enrich_res_neg$log_p, decreasing = T),]
View(data.frame(1:nrow(enrich_res_neg),enrich_res_neg$term_name, enrich_res_neg$intersection))
enrich_res_neg = enrich_res_neg[c(2, 5,28,29,32,42,46,48,62,65),]

enrich_res_neg = enrich_res_neg[,colnames(enrich_res_neg) %in% c('term_name', 'p_value')]
enrich_res_neg$log_p = -log(enrich_res_neg$p_value)
title = ''

enrich_res_neg$term_name = gsub('metabolic process', 'metabolism',enrich_res_neg$term_name)
enrich_res_neg$term_name[9] = "The citric acid (TCA) cycle"
enrich_res_neg$term_name <- factor(enrich_res_neg$term_name, 
                                   levels =  enrich_res_neg$term_name[length(enrich_res_neg$term_name):1])


title = ''#'stim'#'Male'
ggplot(enrich_res_neg, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity',fill='skyblue',color='grey10')+
  xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  scale_fill_manual(values = c('skyblue'))+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))



varimax_loading_df_ord = varimax_df[order(varimax_df$F18, decreasing = T),]
varimax_loading_df_ord = data.frame(genes=varimax_loading_df_ord$X,factor=varimax_loading_df_ord$F18)
tail(varimax_loading_df_ord,20) %>% map_df(rev)

varimax_loading_vis = rbind(head(varimax_loading_df_ord,20),tail(varimax_loading_df_ord,20))
varimax_loading_vis$genes <- factor(varimax_loading_vis$genes, levels=varimax_loading_vis$genes)
ggplot(varimax_loading_vis,aes(x=genes, y=factor, color=factor))+geom_point(size=2,alpha=1.2)+theme_bw()+
  theme(axis.text.x = element_text(color = "grey20", size = 11.5, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 12, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 14, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 14, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(hjust = 1,angle = 0),
        legend.position="left", legend.direction="vertical")+
  scale_color_gradient(name='Factor\nScore')+
  #scale_colour_gradientn(colours=c("red", "blue"))+
  scale_color_gradient2(name='',midpoint = 0, low = "deepskyblue2", mid = "white",
                        high = "deeppink3", space = "Lab" )+
  ylab('Factor score')+xlab('')


#################################################################################
########################    stimulated PBMC Map ###########################
####################################################################################
varimax_df = read.csv('~/sciFA/Results/varimax_loading_df_lupusPBMC.csv')
df = data.frame(gene= varimax_df$X,factor=varimax_df$F2)
df_pos = df[order(df$factor, decreasing = T),]
num_genes = 200

table_to_vis = df_pos[1:20,]
rownames(table_to_vis) = NULL
colnames(table_to_vis) = c('Gene', 'Score')
table_to_vis$Score = round(table_to_vis$Score, 3)
dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(table_to_vis, theme=tt2)

enrich_res = get_gprofiler_enrich(markers=df_pos$gene[1:num_genes], 
                                  model_animal_name = model_animal_name )#'gp__SEA8_T0ld_VHU'
enrich_res_pos = data.frame(enrich_res$result)
enrich_res_pos = enrich_res_pos[,!colnames(enrich_res_pos)%in%'evidence_codes']
enrich_res_pos$log_p = -log(as.numeric(enrich_res_pos$p_value))
enrich_res_pos = enrich_res_pos[order(enrich_res_pos$log_p, decreasing = T),]
View(data.frame(1:nrow(enrich_res_pos),enrich_res_pos$term_name, enrich_res_pos$intersection))

#enrich_res_pos = enrich_res_pos[c(1, 2, 12, 13, 16,19,25, 31, 33, 43),] ## positive - F9 - PBMC dataset
enrich_res_pos = enrich_res_pos[c(1,2,7,11,13, 16,25,31, 36,39),]## positive - F2 - PBMC dataset
enrich_res_pos = enrich_res_pos[,colnames(enrich_res_pos) %in% c('term_name', 'log_p')]

enrich_res_pos$term_name = gsub('metabolic process', 'metabolism',enrich_res_pos$term_name)
enrich_res_pos$term_name <- factor(enrich_res_pos$term_name, levels =  enrich_res_pos$term_name[length(enrich_res_pos$term_name):1])

title = ''#'stim'#'Male'
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity',fill='grey80',color='grey3')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  scale_fill_manual(values = c('grey80'))+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 14, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))



####################################################################################
varimax_df = read.csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv')
genes = read.csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv')
df = data.frame(genes= genes$X0,factor=factor_loading$X28)
varimax_loading_df_ord = df[order(df$factor, decreasing = F),]



##### Figure-1 example figure
varimax_loading_vis = data.frame(genes=paste0('Gene ',1:10),factor=10:1/10+rnorm(n=10,mean = 0,sd = 0.02))

varimax_loading_vis$genes <- factor(varimax_loading_vis$genes, levels=varimax_loading_vis$genes)
ggplot(varimax_loading_vis,aes(x=genes, y=factor, color=factor))+geom_point(size=6,alpha=1.2)+theme_bw()+
  theme(axis.text.x = element_text(color = "grey20", size = 22, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 12, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 14, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 25, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(hjust = 1,angle = 0),
        legend.position="left", legend.direction="vertical")+
  scale_color_gradient(name='Factor\nLoading')+
  #scale_colour_gradientn(colours=c("red", "blue"))+
  scale_color_gradient2(name='',midpoint = 0, low = "deepskyblue2", mid = "white",
                        high = "midnightblue", space = "Lab" )+
  ylab('Factor\nLoading')+xlab('')



enrich_res_pos = data.frame(term_name=paste0('Pathway ',1:8),log_p=8:1/8+rnorm(n=8,mean = 0,sd = 0.05))
enrich_res_pos$factor[1:4] = enrich_res_pos$factor[1:4]+0.6

enrich_res_pos$term_name <- factor(enrich_res_pos$term_name, 
                                   levels =  enrich_res_pos$term_name[length(enrich_res_pos$term_name):1])


title = ''#'stim'#'Male'
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity',fill='lightskyblue3',color='grey3')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  scale_fill_manual(values = c('grey80'))+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 18, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 23, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))

