# Mutational Meltdown in Asexual Populations Doomed to Extinction: Code and Data 

Peter Olofsson, Logan Chipkin, Ryan C. Daileda, Ricardo B. R. Azevedo

Submitted to *Journal of Mathematical Biology* (February 10, 2023).  [Available
in bioRxiv.](https://www.biorxiv.org/content/10.1101/448563v3)

## Contents

* `README.md`: This file.
* [`data`](data): Folder containing data to generate figures.
    * [`pars.csv`](data/pars.csv): empirical estimates of mutational parameters used to generate
      figure 1.
    * `hist.npy`: extinction times from stochastic simulations under default mutational parameters. 
    * `heatA.npy`: data for heat maps in figure 5 (mean extinction time).
    * `heatB.npy`: data for heat maps in figure 5 (variance in extinction time).
    * `heatC.npy`: data for heat maps in figure 5 (time to first click).
* [`figures`](figures): Folder containing all figures.
    * figure 1: [`pars.pdf`](figures/pars.pdf)
    * figure 2: [`branch.pdf`](figures/branch.pdf) (and [TeX source](figures/branch.tex))
    * figure 3: [`N.pdf`](figures/N.pdf)
    * figure 4: [`hist.pdf`](figures/hist.pdf)
    * figure 5: [`heat.pdf`](figures/heat.pdf)
* [`python`](python): Folder containing code and data to generate figures.
    * [`doomed.py`](python/doomed.py): Python 3.7 code to do numerical calculations and run simulations.
    * [`pars.ipynb`](python/pars.ipynb): Jupyter notebook to generate [figure 1](figures/pars.pdf).
    * [`N.ipynb`](python/N.ipynb): Jupyter notebook to generate [figure 3](figures/N.pdf).
    * [`hist.ipynb`](python/hist.ipynb): Jupyter notebook to generate [figure 4](figures/hist.pdf).
    * [`heat.ipynb`](python/heat.ipynb): Jupyter notebook to generate [figure 5](figures/heat.pdf).