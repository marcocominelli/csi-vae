# csi-vae

This repository hosts the code used in the paper:
```
Marco Cominelli, Francesco Gringoli, Lance M. Kaplan, Mani B. Srivastava, Federico Cerutti,
"Accurate Passive Radar via an Uncertainty-Aware Fusion of Wi-Fi Sensing Data"
26th International Conference on Information Fusion (FUSION 2023)
```
and in an extended version of the work currently submitted to IEEE Transactions on Mobile Communications.

There are three Jupyter notebooks in this repository:

- **CSI-VAE.ipynb** contains the code used to train the Variational Auto-Encoders presented in the papers;
- **CSI-EDL-MLP.ipynb** contains the code used to evaluate the performance of different MLP/EDL architectures, and to train a simple decision tree model for the FUSION paper;
- **classifiers-comparison.ipynb** contains the code used to compare the performance of different classifiers in the extension of the original work.

## Useful links

- [Raw CSI dataset](https://doi.org/10.5281/zenodo.7732595)
- [Pre-processed dataset and models](https://doi.org/10.5281/zenodo.7983057) used in the FUSION 2023 paper.
- [Pre-processed dataset and models](https://doi.org/10.5281/zenodo.8239343) used in the work submitted to IEEE Transactions of Mobile Computing (currently under review).
