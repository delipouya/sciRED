from setuptools import setup

VERSION = '0.3.3' 
DESCRIPTION = 'single cell interpretable Residual Decomposition'
LONG_DESCRIPTION = "sciRED is a Python package designed to improve the interpretation of single-cell RNA sequencing data, specifically focusing on signal extraction via factor decomposition. It simplifies the process by removing confounding effects, mapping factors to covariates, identifying unexplained factors, and annotating genes and biological processes. Applying sciRED to various scRNA-seq datasets can unveil diverse biological signals, such as health/disease variation, cell-type identity, sex/age differences, stimulation signals, and rare cell type signatures."

setup(
    name='sciRED',
    url='https://github.com/delipouya/sciRED.git',
    author='Delaram Pouyabahar',
    author_email='d.pouyabahar@mail.utoronto.ca',
    packages=['sciRED'],
    
    # Needed for dependencies
    install_requires=['numpy','pandas','scanpy','statsmodels','seaborn','umap-learn', 'matplotlib',
                      'scikit-learn','scipy','xgboost','scikit-image','diptest==0.2.0'],
    extras_require={
        'dev': ['pytest', 'twine']
    },
    version=VERSION,
    
    license='MIT',
    description=DESCRIPTION,
    keywords=['sciRED', 'single cell RNA-seq', 'interpretability', 'factor decomposition'],

    long_description=LONG_DESCRIPTION,

    classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)


