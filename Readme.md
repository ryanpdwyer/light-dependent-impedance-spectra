# Light-dependent impedance spectra theory and numerical simulations

This repository contains the Python code to reproduce the numerical analysis from the paper "Light-dependent impedance spectra and transient photoconductivity in a Ruddlesden–Popper 2D lead-halide perovskite revealed by electrical scanned probe microscopy and accompanying theory," by Tirmzi, et al.

## Installation

    conda create -n light-dependent-impedance-spectra python=3 numpy scipy matplotlib pandas
    conda activate light-dependent-impedance-spectra
    pip install ray json_tricks tqdm altair altair_saver


## Notes

- Scipy added a warning (version 1.4.1) when a weighted integral is used and points are specified as well (see https://github.com/scipy/scipy/pull/10912); the points are ignored. This does not affect the analysis, but does generate a large number of warning messages!
