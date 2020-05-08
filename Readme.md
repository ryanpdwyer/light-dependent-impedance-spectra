# Light-dependent impedance spectra theory and numerical simulations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3817473.svg)](https://doi.org/10.5281/zenodo.3817473)

This repository contains the Python code to reproduce the numerical analysis from the paper "Light-dependent impedance spectra and transient photoconductivity in a Ruddlesden–Popper 2D lead-halide perovskite revealed by electrical scanned probe microscopy and accompanying theory," by Tirmzi, et al.

## Installation

    conda create -n light-dependent-impedance-spectra python=3 numpy scipy matplotlib pandas
    conda activate light-dependent-impedance-spectra
    pip install ray json_tricks tqdm altair altair_saver


## Code

The file `tdpkefm.py` contains the classes used for numerically simulating the experiments and calculating the frequency shift analytically.


## Reproducing the simulations

The simulations are contained in the python files

    S1-and-3-simulate.py
    S2-simulate.py
    S4-simulate.py

To run the simulations, use `python S1-and-3-simulate.py` using an environment with the dependencies listed above, or on Mac/Linux, use the `run-simulations.sh` script. On a 2019 16" MacBook Pro, all simulations could be completed in less than an hour.

## Plotting the simulations

The Jupyter notebooks `S1-plot.ipynb, S2-plot.ipynb, S3-plot.ipynb, and S4-plot.ipynb` generate the plots from intermediate data files saved in the results folder (plotting dependencies are matplotlib, altair, and altair_saver for saving Fig. S3).