
# Environment Setup

1. Create a new conda environemnt  `conda create --name ml python=3.11`
2. Activate the new environment   `activate ml`
3. Install the packages from requirements.txt `pip install -r requirements.txt`
4. Register the env with jupyter - `python -m ipykernel install --user --name=ml`
5. Access the project folder from dropbox link and save locally , or clone from git
  - https://gatech.box.com/s/jbuy8kwy59chep9ggxam36bkib0v5u23
  - https://github.com/PavanKB/PS_3

# DATA
1. The datasets required for the project will be downloaded directly from UCI Machine Learning repo
2. Dataset urls
    a. https://archive.ics.uci.edu/dataset/20/census+income
    b. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset


# MODELS
1. Each clustering algo is applied for the dataset and saved as seperate python notebook.

* kmeans_dry_beans.ipynb
* expectation_max_dry_bean.ipynb

* kmeans_census_income.ipynb
* expectattion_max_census_income.ipynb

1. Apply the dimensionality reduction on each of the dataset

* pca_dry_beans.ipynb
* ica_dry_bean.ipynb
* rca_dry_bean.ipynb
* manifold_learning_dry_bean.ipynb

* pca_census_income.ipynb
* ica_census_income.ipynb
* rca_census_income.ipynb
* manifold_learning_census_income.ipynb




# RUN THE MODELS
1. Open command prompt and navigate to the project folder
2. Activate the conda env - `activate ml`
3. Start the jupter lab - `jupyter lab`
4. Open the respective the model and dataset python notebook and run the full notebook. It will download the required data and train the model.
5. The trained models are saved as pickle files in ./model
