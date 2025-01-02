library(ggplot2)
library(RColorBrewer)
color_palette <- brewer.pal(n = 12, name = "Set1")

nmf_scores_df_merged = read.csv('/home/delaram/sciRED/review_analysis/NMF_scores_df_merged_lupusPBMC.csv')
nmf_loading_df = read.csv('/home/delaram/sciRED/review_analysis/NMF_loading_df_lupusPBMC.csv')

head(nmf_scores_df_merged)
head(nmf_loading_df)
summary(nmf_loading_df$F30)


factor_number = 23

ggplot(nmf_scores_df_merged, aes(x = F1, y = F23, color = stim)) +
  geom_point(size = 1.6, alpha = 0.6) +  
  scale_color_brewer(palette = "Set1") + 
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "F1 Score",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Stimulation",  # Legend title
    title = paste0("Scatter Plot of F1 vs F", 
                   factor_number,  " Colored by stimulation status")
  )


factor_number = 23
ggplot(nmf_scores_df_merged, aes(x = F1, y = F23, color = cell)) +
  geom_point(size = 1.6, alpha = 0.6) +  
  scale_color_manual(values = color_palette) +  
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "F1 Score",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )


factor_number = 4
ggplot(nmf_scores_df_merged, aes(y = F4, x = cell,fill = cell)) +
  geom_boxplot() +  
  scale_fill_manual(values = color_palette) +  
  theme_classic() +  # Use a clean theme
  theme(
    axis.text.x = element_text(size = 12, angle=90),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )



factor_number = 4
ggplot(nmf_scores_df_merged, aes(y = F4, x = stim,fill = stim)) +
  geom_boxplot() +  # Adjust point size and transparency
  scale_fill_brewer(palette = "Set1") +  # Use the custom Set3 color palette
  theme_classic() +  # Use a clean theme
  theme(
    axis.text.x = element_text(size = 16),  
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14), 
    axis.title.y = element_text(size = 14), 
    legend.title = element_text(size = 13), 
    legend.text = element_text(size = 11),  
  ) +
  labs(
    x = "",  # Label for x-axis
    y = paste0("F",factor_number," Score"),  # Label for y-axis
    color = "Cell Type",  # Legend title
    #title = paste0("Scatter Plot of F1 vs F", factor_number,  " Colored by cell type")
  )


factor_number=21
colnames(nmf_loading_df)[factor_number+1]
nmf_df_ord = data.frame(Genes=nmf_loading_df$X, Loading=nmf_loading_df[,factor_number+1])
nmf_df_ord = nmf_df_ord[order(nmf_df_ord$Loading, decreasing = T),]
nmf_df_ord_vis = head(nmf_df_ord,20)
nmf_df_ord_vis$Genes <- sub("-EN.*", "", nmf_df_ord_vis$Genes)

nmf_df_ord_vis$Genes <- factor(nmf_df_ord_vis$Genes, levels=nmf_df_ord_vis$Genes)
ggplot(nmf_df_ord_vis,aes(x=Genes, y=Loading, color=Loading))+geom_point(size=2,alpha=1.2)+theme_bw()+
  theme(axis.text.x = element_text(color = "grey20", size = 11.5, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 12, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 14, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 14, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(hjust = 1,angle = 0),
        legend.position="left", legend.direction="vertical")+
  scale_color_gradient(name='Factor\nScore')+
  scale_colour_gradientn(colours=c("deepskyblue", "darkblue"))+
  ylab('Factor score')+xlab('')

