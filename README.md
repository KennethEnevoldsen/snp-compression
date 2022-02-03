SNP-Compression
==============================

This project seeks to compress snp's using deep neural networks.


## Plan of action
- [x] Tranpose and save chromosome 1-22 to the processed datafolder
- [ ] Train a model to compress chromosome 22 (shortest chromosome)
- [ ] Train a model to compress chromosome 6 (include the highly correlated region MHC)



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── interim        <- The .zarr files created using src/data/plink_to_dask.py
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   ├── slurm-output   <- slurm .out files
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py

