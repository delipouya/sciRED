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

            


def plot_umap(pca_scores, cell_color_vec, legend_handles=False,
                   plt_legend_list = None,
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
    if legend_handles:
        if len(plt_legend_list) > 4: 
            ### make the legend small if there are more than 4 covariates and place it outside the plot
            plt.legend(handles=plt_legend_list, bbox_to_anchor=(1.05, 1), 
            loc='upper left', borderaxespad=0., prop={'size': 6})

        else:
            plt.legend(handles=plt_legend_list)

    plt.show()





def plot_FCAT(fcat_df, title='', color="YlOrBr",x_axis_label=None,
                               x_axis_fontsize=40, y_axis_fontsize=40, title_fontsize=40,
                               x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, figsize_x=None, 
                               legend_fontsize=32,
                               figsize_y=None, save=False, save_path='./file.pdf'):
    '''
    colors: SV: 'RdPu', AUC/: 'YlOrBr', wilcoxon: rocket_r, featureImp: coolwarm
    plot the score of all the factors for all the covariate levels
    fcat_df: a dataframe of a score for all the factors for all the covariate levels
    '''
    if not figsize_x:
        figsize_x = fcat_df.shape[1]+6
    if not figsize_y:
        figsize_y = fcat_df.shape[0]+2
         
    fig, ax = plt.subplots(figsize=(figsize_x,(figsize_y))) ## x axis, y axis
    ax = sns.heatmap(fcat_df, cmap=color, linewidths=.5, annot=False)
    ax.set_title(title)
    ax.set_xlabel('Factors')
    ax.set_ylabel('Covariate level')
    ### increase the fontsize of the x and y ticks axis labels
    ax.xaxis.label.set_size(x_axis_fontsize)
    ax.yaxis.label.set_size(y_axis_fontsize)
    ### increase the fontsize of the x and y ticks
    
    ### set title fontsize
    ax.title.set_size(title_fontsize)

    ## add F1, F2, ... to the x-axis ticks
    ### if x_axis_label is not None, use x_axis_label as the x-axis ticks
    if x_axis_label is None:
         x_axis_label = ['F'+str(i) for i in range(1, fcat_df.shape[1]+1)]
    
    ax.set_xticklabels(x_axis_label, rotation=45, ha="right",)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis='x', labelsize=x_axis_tick_fontsize)
    ax.tick_params(axis='y', labelsize=y_axis_tick_fontsize)


    ### increase the legend fontsize and make teh legened bar smaller
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize)
    cbar.ax.yaxis.label.set_size(legend_fontsize)
    plt.show()

    if save:
        fig.savefig(save_path, bbox_inches='tight')




def plot_histogram(values, xlabel='scores',title='', bins=100,threshold=None, 
                   save=False, save_path='./file.pdf',
                   xlabel_fontsize=20, ylabel_fontsize=20, title_fontsize=20,
                   xticks_fontsize=16,yticks_fontsize=16 ) -> None:
      '''
      plot the histogram of a list of values
      values: list of values
      xlabel: xlabel of the histogram
      title: title of the histogram
      '''
      plt.figure(figsize=(10, 5))
      plt.hist(values, bins=bins)
      ## adding a line for a threshold value
      if threshold:
            plt.axvline(x=threshold, color='red')
      plt.xlabel(xlabel)
      plt.ylabel('Frequency')
      plt.title(title)
      ### increase the fontsize of the x and y ticks
      plt.xticks(fontsize=xticks_fontsize)
      plt.yticks(fontsize= yticks_fontsize)
      ### increase the fontsize of the x and y labels
      plt.xlabel(xlabel, fontsize= xlabel_fontsize)
      plt.ylabel('Frequency', fontsize= ylabel_fontsize)
      plt.title(title, fontsize= title_fontsize)
      if save:
            plt.savefig(save_path, bbox_inches='tight')

      plt.show()



def plot_matched_factor_dist(matched_factor_dist, title='', save=False, save_path='./file.pdf'):
      '''
        plot the distribution of the number of matched covariate levels for each factor
        matched_factor_dist: the distribution of the number of matched covariate levels for each factor
        title: the title of the plot
        save: whether to save the plot
        save_path: the path to save the plot
      '''
      plt.figure(figsize=(np.round(len(matched_factor_dist)/3),4))
      plt.bar(np.arange(len(matched_factor_dist)), matched_factor_dist)
      ### add F1, F2, ... to the xticks
      plt.xticks(np.arange(len(matched_factor_dist)), 
                 ['F'+str(i) for i in range(1, len(matched_factor_dist)+1)])
      ### make the xticks vertical and set the fontsize to 14
      plt.xticks(rotation=90, fontsize=18)
      #plt.xlabel('Number of matched covariates')
      ## set y ticks as digits and remove the decimal points and half points
      plt.yticks(np.arange(0, max(matched_factor_dist)+1, 1)) 
      plt.ylabel('Number of matched covariate levels', fontsize=18)
      plt.title(title)
      
      if save:
        plt.savefig(save_path, bbox_inches='tight')
      plt.show()


def plot_matched_covariate_dist(matched_covariate_dist, covariate_levels , title='',
                                save=False, save_path='./file.pdf'):
      """
        plot the distribution of the number of matched factors for each covariate level
        matched_covariate_dist: the distribution of the number of matched factors for each covariate level
        covariate_levels: the covariate levels
        title: the title of the plot
        save: whether to save the plot
        save_path: the path to save the plot

      """
      plt.figure(figsize=(np.round(len(matched_covariate_dist)/3),4))
      plt.bar(np.arange(len(matched_covariate_dist)), matched_covariate_dist)
      ### add covariate levels to the xticks
      plt.xticks(np.arange(len(matched_covariate_dist)), covariate_levels)

      ### make the xticks vertical and set the fontsize to 14
      plt.xticks(rotation=90, fontsize=18)
      #plt.xlabel('Number of matched factors')
      ## set y ticks as digits and remove the decimal points and half points
      plt.yticks(np.arange(0, max(matched_covariate_dist)+1, 1), fontsize=19) 
      plt.ylabel('Number of matched factors', fontsize=18)
      plt.title(title)
      
      if save:
        plt.savefig(save_path, bbox_inches='tight')
      plt.show()




def plot_factor_cor_barplot(factor_libsize_correlation, 
                            x_tick_labels=None, title='', y_label='', x_label=''):
    plt.figure(figsize=(15,5))
    if x_tick_labels is None:
        ### set the x axis labels as F1, F2, ...
        x_tick_labels = ['F'+str(i+1) for i in range(factor_libsize_correlation.shape[0])]
    ### set background color to white
    plt.rcParams['axes.facecolor'] = 'white'
    ## add y and x black axis 
    plt.axhline(y=0, color='black', linewidth=2)
    plt.bar(x_tick_labels, factor_libsize_correlation, color='black')
    ## set y range from 0 to 1
    plt.ylim(-0.5,1)
    plt.xticks(fontsize=25, rotation=90)
    plt.yticks(fontsize=25)
    plt.xlabel(x_label, fontsize=28)
    plt.ylabel(y_label, fontsize=28)
    plt.title(title, fontsize=26)
    plt.show()
