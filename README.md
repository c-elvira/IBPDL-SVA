# IBPDL-SVA
[![Build Status](https://travis-ci.com/c-elvira/IBPDL-SVA.svg?branch=master)](https://travis-ci.com/c-elvira/IBPDL-SVA)
[![CodeFactor](https://www.codefactor.io/repository/github/c-elvira/ibpdl-sva/badge)](https://www.codefactor.io/repository/github/c-elvira/ibpdl-sva)

This repository implements the IBP-DL SVA algorithm from [dang2018,elvira2018].

## Prerequirements

Our instructions have been tested on Linux and Mac only.
On Mac, you may need to install a compiler, *e.g.*, *gcc* (as part of the XCode command line tools).

## Install from sources

1. Clone this repository

```
git clone https://github.com/c-elvira/IBPDL-SVA.git
cd IBPDL-SVA
```

And execute `setup.py` using,  e.g. using `pip

```
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
```

Or simply run
```
./install.sh
```

2. Test

The installation can be tested using the following simple command
```
python setup.py test
```


## Usage examples

Work in Progress :)

## Reproducible research

The folders 'exp_eusipco' and 'exp_cap' contain the code to reproduce the experiments in  [dang2018,elvira2018].


## This work is associated to the following papers

``` latex
@Inproceedings{elvira2018,
    title     = {Small variance asymptotics and bayesian nonparametrics for dictionary learning},
    author    = {Elvira, Clément and Dang, Hong-Phuong and Chainais, Pierre},
    booktitle = {Proc. European Signal Processing Conf. (EUSIPCO)},
    address   = {Rome, Italy},
    year      = {2018},
    month     = {Sept.},
}

@Inproceedings{dang2018,
    title     = {Vers une méthode d'optimisation non paramétrique pour l'apprentissage de dictionnaire en utilisant Small-Variance Asymptotics pour modèle probabiliste},
    author    = {Dang, Hong-Phuong and Elvira, Clément and Chainais, Pierre},
    booktitle = {(CAP)},
    address   = {Rouen, France},
    year      = {2018},
    month     = {Juin},
}
```
