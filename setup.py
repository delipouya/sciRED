from setuptools import setup


VERSION = '0.0.1' 
DESCRIPTION = 'single cell Interpretable Residual Decomposition'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sciRED',
    url='https://github.com/delipouya/sciRED.git',
    author='Delaram Pouyabahar',
    author_email='d.pouyabahar@mail.utoronto.ca',
    packages=['sciRED'],
    # Needed for dependencies
    install_requires=['numpy','pandas','scanpy','statsmodels','seaborn','umap', 'matplotlib',
                      'scikit-learn','scipy','random','xgboost','time','skimage','diptest'],
    extras_require={
        'dev': ['pytest', 'twine']
    },

    version=VERSION,
    
    license='MIT',
    description=DESCRIPTION,
    keywords=['sciRED', 'single cell RNA-seq'],

    long_description=long_description,

    classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)


