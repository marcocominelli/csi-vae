# csi-vae

This repository hosts the code used in the following papers:

- Marco Cominelli, Francesco Gringoli, Lance M. Kaplan, Mani B. Srivastava, Trevor Bihl, Erik P. Blasch, Nandini Iyer, and Federico Cerutti,
"Neuro-Symbolic Fusion of Wi-Fi Sensing Data for Passive Radar with Inter-Modal Knowledge Transfer",
27th International Conference on Information Fusion (FUSION 2024)


- Marco Cominelli, Francesco Gringoli, Lance M. Kaplan, Mani B. Srivastava, Federico Cerutti,
[Accurate Passive Radar via an Uncertainty-Aware Fusion of Wi-Fi Sensing Data](https://ieeexplore.ieee.org/document/10224098),
26th International Conference on Information Fusion (FUSION 2023)

The code is provided under an MIT license.
Please consider citing the above works if you use the code in this repository.

## Source code

There are several Python scripts and Jupyter notebooks in this repository.
They have been used in two different research works presented at [FUSION 2023](https://fusion2023.org) and [FUSION 2024](https://fusion2024.org).

### FUSION 2024
- **deepprobhar.py** contains the code used to train and test DeepProbHAR, a neuro-symbolic architecture for human activity recognition using Wi-Fi sensing data;
- **har.pl** contains the DeepProbLog code of DeepProbHAR;
- **get_deepprobhar_mlp_accuracy.py** contains the code used to measure the accuracy of single MLPs in the DeepProbHAR neuro-symbolic architecture;
- **independent_mlps.py** contains the code used to train several MLPs on finely labelled dataset comprising only a subset of the activities;
- **comparison_single_mlp.py** contains the code used to compare the results of DeepProbHAR against a single MLP substituting the entire neuro-symbolic architecture.

**How to run the code:**
It is advised to install all the required packages in a new Conda environment:
```
  $ conda create --name fusion2024 python=3.10
  $ conda activate fusion2024
```

Then, you can use `pip` to install the required packages:
```
  $ pip install -r requirements.txt
```

The code has been tested on an Ubuntu 22 distribution using Python 3.10.14.


### FUSION 2023
- **CSI-VAE.ipynb** contains the code used to train the Variational Auto-Encoders presented in the papers;
- **CSI-EDL-MLP.ipynb** contains the code used to evaluate the performance of different MLP/EDL architectures, and to train a simple decision tree model;
- **classifiers-comparison.ipynb** contains the code used to compare the performance of different classifiers in an extension of the work.

## Other useful links

- [Raw CSI dataset](https://doi.org/10.5281/zenodo.7732595)
- [Pre-processed dataset and models](https://doi.org/10.5281/zenodo.11367111) used in the FUSION 2024 paper.
- [Pre-processed dataset and models](https://doi.org/10.5281/zenodo.7983057) used in the FUSION 2023 paper.
- [Pre-processed dataset and models](https://doi.org/10.5281/zenodo.8239343) used to extend the work in the FUSION 2023 paper.