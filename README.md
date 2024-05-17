
<img src="https://github.com/delipouya/sciRED/blob/main/inst/sciRED_logo_wave.png" align="right" height="200">

# sciRED

- [Introduction](#introduction)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Citation](#citation)

## Introduction

Single-cell RNA sequencing (scRNA-seq) maps gene expression heterogeneity within a tissue. However, identifying biological signals in this data is challenging due to confounding technical factors, noise, sparsity, and high dimensionality. Data factorization methods address this by separating and identifying signals, such as gene expression programs, in the data, but the resulting factors must be manually interpreted. We developed sciRED as a tool to improve the interpretation of scRNA-seq factor analysis. sciRED has four steps: 1) removing known confounding effects and using rotations to improve factor interpretability; 2) mapping factors to known covariates; 3) identifying unexplained factors that may capture hidden biological phenomena; and 4) determining the genes and biological processes represented by the resulting factors. We apply sciRED to multiple scRNA-seq data sets and identify sex-specific variation in a kidney map, discern strong and weak stimulation signals in a PBMC dataset, reduce ambient RNA contamination in a rat liver atlas to help identify strain variation, and reveal rare cell type signatures and anatomical zonation gene programs in a healthy human liver map. These demonstrate that sciRED is useful in characterizing diverse biological signals within scRNA-seq datasets.


## Installation
Please make sure to install the following packages before installing sciRED:
numpy, pandas, scanpy, statsmodels, seaborn, umap-learn, matplotlib, scikit-learn, scipy, xgboost, scikit-image, [diptest==0.2.0](https://pypi.org/project/diptest/0.2.0/)


## Tutorial

Follow [this link](https://github.com/delipouya/sciRED/blob/main/tutorial_stimulatedPBMC.ipynb) to
learn how to use sciRED. The tutorial introduces the standard processing
pipeline and applies it to a stimulated PBMC dataset.

## Citation

If you find sciRED useful for your publication, please cite:
[Pouyabahar et al. Interpretable single-cell factor decomposition using sciRED.](url)