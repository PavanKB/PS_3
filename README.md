
# Environment Setup

1. Create a new conda environemnt  `conda create --name ml python=3.11`
2. Activate the new environment   `activate ml`
3. Install the packages from requirements.txt `pip install -r requirements.txt`
4. Register the env with jupyter - `python -m ipykernel install --user --name=ml`
5. Access the project folder from dropbox link and save locally , or clone from git
  - https://gatech.box.com/s/9qwr6o2lojc79zsac38aku1hmar0w9q5
  - https://github.com/PavanKB/PS_3

# DATA
1. The datasets required for the project will be downloaded directly from UCI Machine Learning repo
2. Dataset urls
    a. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
    b. https://archive.ics.uci.edu/dataset/186/wine+quality


# MODELS

## EDA

* eda_dry_bean_dataset.ipynb
* eda_wine_quality.ipynb

## Clustering
Each clustering algo is applied for the dataset and saved as seperate python notebook.

* kmeans_dry_beans.ipynb
* exp_max_dry_beans.ipynb

* kmeans_wine_quality.ipynb
* exp_max_wine_quality.ipynb

## Dimension Reduction
Apply the dimensionality reduction on each of the dataset

* dim_red_dry_beans.ipynb
* dim_red_wine_quality.ipynb

## Dimension Reduction followed by Clustering
Apply dimensionality reduction followed by clustering

* dim_red_kmeans_dry_beans.ipynb
* dim_red_exp_max_dry_beans.ipynb

* dim_red_exp_max_wine_quality.ipynb
* dim_red_kmeans_wine_quality.ipynb

## Neural Network with Dimension Reduction
1. Neural network on Dry Bean dataset with dim reduction

* nn_dry_bean.ipynb
* pca_nn_dry_bean.ipynb
* ica_nn_dry_bean.ipynb
* gaus_proj_nn_dry_bean.ipynb
* tsne_nn_dry_bean.ipynb

## Neural Network with Clustering
* kmeans_nn_dry_beans.ipynb
* exp_max_nn_dry_beans.ipynb

# RUN THE MODELS
1. Open command prompt and navigate to the project folder
2. Activate the conda env - `activate ml`
3. Start the jupter lab - `jupyter lab`
4. Open the respective the model and dataset python notebook and run the full notebook. It will download the required data and train the model.
