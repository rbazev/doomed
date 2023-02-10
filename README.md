# Mutational meltdown in asexual populations doomed to extinction: Code and Data 

Peter Olofsson, Logan Chipkin, Ryan C. Daileda, Ricardo B. R. Azevedo

Submitted to *Journal of Mathematical Biology* (February 10, 2023).  [Available
in bioRxiv](https://www.biorxiv.org/content/10.1101/448563v3)

## Contents

* `README.md`: This file.
* `data`: Folder containing data to generate figures.
    * `pars.csv`: empirical estimates of mutational parameters used to generate
      figure 1.
* `figures`: Folder containing all figures.
    * `figure 1`: pars.pdf
    * `figure 2`: branch.pdf (and TeX source)
    * `figure 3`: N.pdf
    * `figure 4`: hist.pdf
    * `figure 5`: heat.pdf
* `python`: Folder containing code and data to generate figures.
    * `doomed.py`: Python 3.7 code to do numerical calculations and run simulations.
    * `pars.ipynb`: Jupyter notebook to generate `figures/pars.pdf` (figure 1)
    * `N.ipynb`: Jupyter notebook to generate `figures/N.pdf` (figure 3)
    * `hist.ipynb`: Jupyter notebook to generate `figures/hist.pdf` (figure 4)
    * `heat.ipynb`: Jupyter notebook to generate `figures/heat.pdf` (figure 5)