#!/usr/bin/env python

import time,os,re,csv,sys,uuid,joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
plt.style.use('seaborn')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

DATA_DIR = os.path.join(".","data")
IMAGE_DIR = os.path.join(".","images")

def save_fig(fig_id, tight_layout=True, image_path=IMAGE_DIR):
    """
    save the image as png file in the image directory
    """
    
    ## Check the data directory
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    path = os.path.join(image_path, fig_id + ".png")
    print("...saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def ingest_data(datadir=DATA_DIR, filename="tweets.csv"):
    """
    ingest tweets dataset
    """
    
    print("Ingesting data")
    
    ## load csv file from data directory
    tweets = pd.read_csv(os.path.join(datadir,filename))
    
    ## dataframe structure
    print("...dataset of {} rows and {} columns".format(tweets.shape[0], tweets.shape[1]))
        
    ## check duplicates
    is_duplicate = tweets.duplicated(subset=["id"])
    print("...number of duplicates:", len(tweets[is_duplicate]))
    
    ## check missing data
    total = tweets.isnull().sum().sort_values(ascending=False)
    percent = (tweets.isnull().sum()/tweets.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("...missing values: \n {}".format(missing_data.head()))
    
    return tweets

def summarize_data(corpus, preprocessing=True):
    """
    print statements and visualizations to summarize the corpus
    """
    
    print("Summarize data")
    
    # get the documents size
    df_doc_size = pd.Series([len(str(doc).split(" ")) for doc in corpus])
    
    # get the tokens in the corpus
    df_tokens = pd.Series([token for doc in corpus for token in str(doc).split(" ")])
    
    print("---------------------------")
    print("num docs", len(corpus))
    print("median tokens", df_doc_size.median())
    print("num tokens", len(df_tokens))
    print("unique tokens", len(df_tokens.value_counts()))
    print("---------------------------")
    
    # make plots
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    sns.distplot(df_doc_size, ax=ax1)
    ax1.set_title("Document Sizes")
    
    sns.distplot(df_tokens.value_counts().values, ax=ax2)
    ax2.set_title("Tokens Counts")
    
    if preprocessing:
        save_fig("summarize_data_preprocessing")
    else:
        save_fig("summarize_data_postprocessing")
    
if __name__ == "__main__":
    
    run_start = time.time()
    
    ## load tweets
    tweets = ingest_data()
    
    ## summarize data
    summarize_data(tweets.text.values.tolist())
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
