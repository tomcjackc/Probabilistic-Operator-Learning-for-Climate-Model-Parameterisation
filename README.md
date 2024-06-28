# Probabilistic Operator Learning for Climate Model Parameterisation

This repository contains the code required to reproduce the results presented in the MRes Thesis "Probabilistic Operator Learning for Climate Model Parameterisation", submitted at The University of Cambridge.

# Setup

We recommend first cloning this repository locally. Navigate to your chosen directory and use

    `git clone https://github.com/tomcjackc/Probabilistic-Operator-Learning-for-Climate-Model-Parameterisation.git`

The `conda` environment used to conduct this work can be created by navigating using

    `conda create --name <env-name> --file requirements.txt`

Notebooks in the `notebooks/` diectory should now be runnable. If the `import` statements raise exceptions, please contact [tc656@cam.ac.uk](mailto:tc656@cam.ac.uk) and we'll be happy to help.

# Data download

Data for the benchmark problems --- Burgers', Darcy Flow, Helmholtz, Navier-Stokes --- can be obtained from  https://doi.org/10.5281/zenodo.12529654. Data for each of these problems (details found on the Zenodo repository) should be placed into the `burgers_data/`, `darcy_flow_data/rect_cont_PWC/`, `helmholtz_data/`, and `navier_stokes_data/` directories, respectively.

# Notebooks

