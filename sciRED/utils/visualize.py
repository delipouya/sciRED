import seaborn as sns
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import random


def plot_pca(pca_scores, 
                   num_components_to_plot, 
                   cell_color_vec, 
                   legend_handles=False,
                   plt_legend_list = None,
                   title='PCA of the data matrix') -> None:
    '''
    plot the PCA components with PC1 on the x-axis and other PCs on the y-axis
    pca_scores: the PCA scores for all the cells
    num_components_to_plot: the number of PCA components to plot as the y-axis
    cell_color_vec: the color vector for each cell
    legend_handles: whether to show the legend handles
    plt_legend_list: a list of legend handles for each covariate
    title: the title of the plot
    '''
    
    for i in range(1, num_components_to_plot):
        ## color PCA based on strain
        plt.figure()
        ### makke the background white with black axes
        plt.rcParams['axes.facecolor'] = 'white'
        
        plt.scatter(pca_scores[:,0], pca_scores[:,i], c=cell_color_vec, s=1) 
        plt.xlabel('F1')
        plt.ylabel('F'+str(i+1))
        plt.title(title)
        if legend_handles:
            if len(plt_legend_list) > 4: 
                ### make the legend small if there are more than 4 covariates and place it outside the plot
                plt.legend(handles=plt_legend_list, bbox_to_anchor=(1.05, 1), 
                loc='upper left', borderaxespad=0., prop={'size': 6})

            else:
                plt.legend(handles=plt_legend_list)



        
        
        plt.show()



def plot_factor_scatter(factor_scores, x_i, y_i, 
                        cell_color_vec, covariate=None,plt_legend_dict=None,
                        title='') -> None:
    '''
    plot the scatter plot of two factors
    factor_scores: the factor scores for all cells
    x_i: the index of the x-axis factor
    y_i: the index of the y-axis factor
    cell_color_vec: the color vector for each cell
    covariate: the covariate to color the cells
    title: the title of the plot
    '''
    plt.figure()
    plt.scatter(factor_scores[:,x_i], factor_scores[:,y_i], c=cell_color_vec, s=1) 
    plt.xlabel('F'+str(x_i+1))
    plt.ylabel('F'+str(y_i+1))
    if covariate and plt_legend_dict:
        plt.legend(handles=plt_legend_dict[covariate])
    plt.title(title)
    plt.show()



def plot_factor_loading(factor_loading, genes, x_i, y_i, fontsize=6, num_gene_labels=5,
                        title='Scatter plot of the loading vectors', label_x=True, label_y=True) -> None:
      '''
      plot the scatter plot of two factors
      factor_loading: the factor loading matrix
      genes: the gene names
      x_i: the index of the x-axis factor
      y_i: the index of the y-axis factor
      fontsize: the fontsize of the gene names
      title: the title of the plot
      label_x: whether to label the x-axis with the gene names
      label_y: whether to label the y-axis
      '''

      plt.figure(figsize=(10, 10))
      plt.scatter(factor_loading[:,x_i], factor_loading[:,y_i], s=10)
      plt.xlabel('F'+str(x_i+1))
      plt.ylabel('F'+str(y_i+1))
      
      if label_y:
            ### identify the top 5 genes with the highest loading for the y-axis factor
            top_genes_y = np.argsort(factor_loading[:,y_i])[::-1][0:num_gene_labels]
            ### identify the gene nnames of the top 5 genes
            top_genes_y_names = genes[top_genes_y]

            bottom_genes_y = np.argsort(factor_loading[:,y_i])[0:num_gene_labels]
            ### identify the gene nnames of the top 5 genes
            bottom_genes_y_names = genes[bottom_genes_y]
                  ### add the top 5 genes with the highest loading for the y-axis factor to the plot make font size smaller
            for i, txt in enumerate(top_genes_y_names):
                  plt.annotate(txt, (factor_loading[top_genes_y[i],x_i], factor_loading[top_genes_y[i],y_i]), fontsize=fontsize)
            ### add the top 5 genes with the least loading for the y-axis factor to the plot make font size smaller
            for i, txt in enumerate(bottom_genes_y_names):
                  plt.annotate(txt, (factor_loading[bottom_genes_y[i],x_i], factor_loading[bottom_genes_y[i],y_i]), fontsize=fontsize)
      
      
      if label_x:
            top_genes_x = np.argsort(factor_loading[:,x_i])[::-1][0:num_gene_labels]
            top_genes_x_names = genes[top_genes_x]
            bottom_genes_x = np.argsort(factor_loading[:,x_i])[0:num_gene_labels]
            bottom_genes_x_names = genes[bottom_genes_x]
            for i, txt in enumerate(top_genes_x_names):
                  plt.annotate(txt, (factor_loading[top_genes_x[i],x_i], factor_loading[top_genes_x[i],y_i]), fontsize=fontsize)
            for i, txt in enumerate(bottom_genes_x_names):
                  plt.annotate(txt, (factor_loading[bottom_genes_x[i],x_i], factor_loading[bottom_genes_x[i],y_i]), fontsize=fontsize)
      
      plt.title(title)
      plt.show()

            


def plot_umap(pca_scores, cell_color_vec, covariate=None,plt_legend_dict=None,
                    title='UMAP of the PC components of the gene expression matrix') -> None:

    ### apply UMAP to teh PCA components
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(pca_scores)
    print('embedding shape: ', embedding.shape)
    
    ### plot the UMAP embedding
    plt.figure()
    plt.rcParams['axes.facecolor'] = 'white'
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cell_color_vec, s=1)
    plt.title(title)
    if covariate and plt_legend_dict:
        plt.legend(handles=plt_legend_dict[covariate])
    
    plt.show()
