# Discovery of Disease-Associated Genomic Features from Single Cell RNA-seq Data using Interpretable Graph Attention Networks
## CPSC 452 Final Project: Ro Malik, Raymond Hou, Tim Li

#### Dependencies
- torch
- torch-geometric
- scikit-learn
- tqdm (convenience)
- plotnine (visualizations)


#### Datasets
We used one dataset, derived from a processed scRNA count matrix from the MultiomeBrain study, which was performed as part of the PsychENCODE consortium and contributed its data to a consortium-wide pool.
Our dataset contained a subset of oligodendrocyte cells from the prefrontal cortex (Brodmann area 9) of the brains of individuals with and without schizophrenia (48 unique individuals, 24 each).

#### Instructions to run
First, the data needs to be downloaded and placed in the directory where the script will be run. The three required files (metacell expression matrix, metadata table, and topological overlap matrix) can be downloaded from https://drive.google.com/drive/folders/1S9Kk8QNi72NPYhUibbgqtWOuU-80BlS4?usp=sharing.
Then, once the dependencies listed above are installed, hyperparameters can be set as desired at the top of main.py, and the file can be run.
