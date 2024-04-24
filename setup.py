from setuptools import setup, find_packages


VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'


setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='sciRED',
    url='https://github.com/delipouya/sciRED.git',
    author='Delaram Pouyabahar',
    author_email='d.pouyabahar@mail.utoronto.ca',
    # Needed to actually package something
    packages=['sciRED'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version=VERSION,
    # The license can be anything you like
    license='MIT',
    description=DESCRIPTION,
    keywords=['python', 'first package'],
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.txt').read(),
    classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X"
        ]
)


