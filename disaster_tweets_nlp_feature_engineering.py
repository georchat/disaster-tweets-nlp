#!/usr/bin/env python

import time,os,re,csv,sys,uuid,joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
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


## imports from process model scripts
from disaster_tweets_nlp_etl import etl
from disaster_tweets_nlp_data_exploration import save_fig
from disaster_tweets_nlp_data_exploration import DATA_DIR

MODEL_DIR = os.path.join(".","models")
SAVED_MODEL = "feature_engineer.joblib"

def engineer_features(valid_size=0.25, rs=42):
    """
    engineer text corpus
    """
    
    ## etl
    train_data, y_train, valid_data, y_valid = etl()
    
    
    ## build transformation pipeline
    n_topics = 150
    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online',
                                          learning_offset=50., random_state=rs)
    
    feature_engineer = Pipeline([('counter', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ("lda",lda_model)])
    
    ## fit on the training set
    feature_engineer.fit(train_data)
    
    ## transform training set
    X_train = feature_engineer.transform(train_data)
    print("training data dimensions {}".format(X_train.shape))
    
    ## transform the validation set
    X_valid = feature_engineer.transform(valid_data)
    
    ## save transformation pipeline
    saved_model = os.path.join(MODEL_DIR,SAVED_MODEL)
    joblib.dump(feature_engineer, saved_model)
    
    
def create_2D_plot():
    """
    use TSNE for 2D data visualization
    """
    
    ## load the data
    train_data, y_train, valid_data, y_valid = etl()
    
    ## load transformation pipeline
    feature_engineer = joblib.load(os.path.join(MODEL_DIR, SAVED_MODEL))
    
    ## tsne dimensionality reduction for visualiazation
    X_count = feature_engineer.named_steps["counter"].transform(train_data)
    tsne = TSNE(n_components=2, random_state=42)
    X_2D = tsne.fit_transform(X_count)
    print(X_2D.shape)
    
    ## create plots
    plt.figure(figsize=(10,8))
    plt.plot(X_2D[:, 0][y_train==1], X_2D[:, 1][y_train==1], "ro", label="pos")
    plt.plot(X_2D[:, 0][y_train==0], X_2D[:, 1][y_train==0], "bo", label="neg")
    plt.axis('off')
    plt.legend(loc="upper left", fontsize=14)
    save_fig("tsne_visualization")
    

if __name__ == "__main__":
    
    run_start = time.time()
    
    ## feature engineering
    engineer_features()
    
    ## create 2D plot
    create_2D_plot()
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
    
