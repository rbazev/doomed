# Mutational Meltdown in Asexual Populations Doomed to Extinction: Code and Data

Peter Olofsson, Logan Chipkin, Ryan C. Daileda, Ricardo B. R. Azevedo. Mutational meltdown in asexual populations doomed to extinction.
*Journal of Mathematical Biology* 87: 88, 2023 ([doi: 10.1007/s00285-023-02019-y](https://doi.org/10.1007/s00285-023-02019-y)).

Last updated: January 10, 2024.

## Contents

* `README.md`: This file.
* [`data`](data): Folder containing data to generate figures.
    * [`pars.csv`](data/pars.csv): empirical estimates of mutational parameters used to generate
      figure 1.
    * `hist.npy`: extinction times from stochastic simulations under default
      mutational parameters used to generate figure 4.
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
    * [`heat.ipynb`](python/heat.ipynb): Jupyter notebook to generate [figure
      5](figures/heat.pdf).

**Note:** python code was last tested with
* python 3.7.12
* numba 0.53.1
* numpy 1.20.3
* matplotlib 3.4.2
* seaborn 0.11.1
* pandas 1.3.5
