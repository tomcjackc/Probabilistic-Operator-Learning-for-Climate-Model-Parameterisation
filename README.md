# Probabilistic Operator Learning for Climate Model Parameterisation

This repository contains the code required to reproduce the results presented in the MRes Thesis "Probabilistic Operator Learning for Climate Model Parameterisation", submitted to The University of Cambridge.

# Setup

We recommend first cloning this repository locally. Navigate to your chosen directory and use

    git clone https://github.com/tomcjackc/Probabilistic-Operator-Learning-for-Climate-Model-Parameterisation.git

The `conda` environment used to conduct this work can be created by navigating using

    conda create --name <env-name> --file requirements.txt

Notebooks in the `notebooks/` diectory should now be runnable. If the `import` statements raise exceptions, please contact [tc656@cam.ac.uk](mailto:tc656@cam.ac.uk) and we'll be happy to help.

# Data download

Data for the benchmark problems --- Burgers', Darcy Flow, Helmholtz, Navier-Stokes --- can be obtained from  https://doi.org/10.5281/zenodo.12529654. Data for each of these problems (details found on the Zenodo repository) should be placed into the `burgers_data/`, `darcy_flow_data/rect_cont_PWC/`, `helmholtz_data/`, and `navier_stokes_data/` directories, respectively.

# Code used for reproduction

The Jupyter notebooks found in the `notebooks/` directory can be used to reproduce the results presented in the thesis. See `notebooks/README.md` for more details.

Model architectures for our operator learning framework are found in `models.py`, including a base class for our overall model and a gaussian process regressor class (written using GPJax) used for our latent representation of the operator in question.

Code used to load data for each benchmark problem and the PV parameterisation experiment is found in `dataloaders.py`. This ensures that data used in the notebooks is in a standardised format.

Lastly, `tools.py` includes some helper functions required to load data for the PV parameterisation experiment. This code is reused from https://github.com/m2lines/pyqg_generative.git.

The submodule `pyqg_parameterisation_benchmarks` is used to generate new data for PV parameterisation in experiments, but is not used in the final results of this work.