from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo 
from itertools import combinations, product
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
import seaborn as sns


def get_data(name):
    """Get the data from the UCI ML Repository

    Args:
        name (str): The name of the dataset to fetch

    Returns:
        pd.DataFrame: The dataset
    """
    dataset = fetch_ucirepo(name)

    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets 
        
    # metadata 
    metadata = dataset.metadata
        
    # variable information 
    variables = dataset.variables

    return X, y, metadata, variables


def calc_matthews_corrcoef(X):
    """
    Calculate the pair wise matthews corr coeff for all the columns.

    returns a matrix 
    """
    corr_coeff = pd.DataFrame(np.nan, columns=X.columns, index=X.columns)
    
    pairs = list(combinations(X.columns, 2))

    for col in X.columns:
        corr_coeff.loc[col, col] = 1.0

    for col_x, col_y in combinations(X.columns, 2):

        corr_coeff.loc[col_x, col_y] = corr_coeff.loc[col_y, col_x] = matthews_corrcoef(X[col_x], X[col_y])

    return corr_coeff
    

def get_df_details(df):
    """Get them details of the dataframe column wise, with more info than df.describe()

    Args:
        df (pd.DataFrame): The dataframe in wide format to be analysed

    Returns:
        pd.DataFrame: Summary of the dataframe
    """
    df_summ = df.describe(include='all').transpose()
    df_summ['nunique'] = df.nunique()
    df_summ['n_nulls'] = df.isnull().sum()
    df_summ['dtype'] = df.dtypes

    return df_summ
    

def plot_distrb(df):
    """
    plot value counts for each column as subplots
    if the column is categorical, use countplot
    if the column is numerical, use distplot

    # https://www.statology.org/seaborn-subplots/

    Args:
        df (pd.DataFrame): DataFrame in wide formatr to plot
    """
    plot_rows = np.ceil(np.sqrt(df.shape[1])).astype(int)

    fig, ax = plt.subplots(plot_rows, plot_rows, figsize=(20, 20))

    axs = list(product(range(plot_rows), range(plot_rows)))
    
    for i in range(df.shape[1]):

        col = df.iloc[:, i]

        if col.dtype == 'object':
            sns.countplot(col, ax=ax[*axs[i]])
        else:
            sns.histplot(col, ax=ax[*axs[i]])
        

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    # This code is sourced from : https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


def calc_reconstruction_error(X, X_recon):
    """
    Calculate the reconstruction error between the original and reconstructed data
    """
    return np.mean(np.sum((X - X_recon)**2, axis=1))

def plot_kmeans_cluter(X, labels, centroid, title):
    """
    Plot the kmeans clusters with the centroids
    X: The data to plot. It should be 2D
    labels: The cluster labels
    centroid: The centroids of the clusters
    title: The title of the plot
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='viridis', ax=ax)
    sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color='black', s=50, marker='o', ax=ax)
    ax.set_title(title)

    return fig, ax