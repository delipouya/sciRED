
<img src="https://github.com/delipouya/sciRED/blob/main/inst/sciRED_logo_wave.png" align="right" height="200">

# sciRED

- [Introduction](#introduction)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

Single-cell RNA sequencing (scRNA-seq) maps gene expression heterogeneity within a tissue. However, identifying biological signals in this data is challenging due to confounding technical factors, noise, sparsity, and high dimensionality. Data factorization methods address this by separating and identifying signals, such as gene expression programs, but the resulting factors require manual interpretation.

We developed sciRED to enhance the interpretation of scRNA-seq factor analysis. sciRED follows four steps:
1. Removing known confounding effects and using rotations to improve factor interpretability.
2. Mapping factors to known covariates.
3. Identifying unexplained factors that may capture hidden biological phenomena.
4. Determining the genes and biological processes represented by the resulting factors.

We applied sciRED to multiple scRNA-seq datasets and demonstrated its utility in:
- Identifying general and cell-type specific covariate-related variations, such as sex-specific variations in a kidney map, discerning strong and weak stimulation signals in a PBMC dataset, and general and cell-type specific strain variation within a rat liver atlas.
- Employing a cluster-free approach to identify and guide the annotation of cell type identity programs.
- Decomposing signal and noise, such as eliminating ambient RNA contamination in a rat liver atlas to unveil strain variations.
- Evaluating unannotated factors to reveal hidden biology in a healthy human liver map, represented by anatomical zonation gene programs, T cell-specific cell cycle signatures, and two rare cell type signatures that were missed in the original study.


## Installation
Please make sure to install the following packages **before installing sciRED**:
numpy, pandas, scanpy, statsmodels, seaborn, [umap-learn](https://pypi.org/project/umap-learn/), matplotlib, scikit-learn, scipy, xgboost, scikit-image, [diptest==0.2.0](https://pypi.org/project/diptest/0.2.0/)


The sciRED package is available for installation via PyPI. Please ensure you install the most up-to-date version.
```bash
pip install sciRED
```

**Common issues**\
Some of the prerequisite packages require the Numba package for parallel implementation. Please install older versions of numpy (such as 1.22.4) in case you encounter the following error:

```bash
Numba needs NumPy 1.24 or less
```
Detailed package version information is provided in the requirements.txt file.

Installation, verified on Linux and Unix systems, takes less than a minute once all dependencies are resolved.

## Tutorial

Follow [tutorial-1](https://github.com/delipouya/sciRED/blob/main/tutorial1_scMixology.ipynb) and [tutorial-2](https://github.com/delipouya/sciRED/blob/main/tutorial2_stimulatedPBMC.ipynb) to learn how to use sciRED. These tutorials introduce the standard processing pipeline and demonstrate the application of sciRED on the scMixology and stimulated PBMC datasets. Further details about the input datasets are available in the manuscript. The data processing scripts are available in the _data_prep_ folder. 

The input data for Tutorial 1 is provided in the "Data" folder. Instructions for downloading and saving data for Tutorial 2 can be found in the `data_stimPBMC.R` script inside the _data_prep_ folder. The demo for Tutorial 1 took less than a minute to run on a workstation with 24 CPUs and 64 GB RAM.

## Citation

If you find sciRED useful for your publication, please cite:

[Pouyabahar et al. Interpretable single-cell factor decomposition using sciRED.](https://www.biorxiv.org/content/10.1101/2024.08.01.605536v2)

For the code and Zenodo archive:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14595050.svg)](https://doi.org/10.5281/zenodo.14595050)

## Contact
For questions, please contact Delaram Pouyabahar at: d.pouyabahar@mail.utoronto.ca
