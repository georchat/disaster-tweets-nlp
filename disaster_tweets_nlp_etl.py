#!/usr/bin/env python

import time,os,re,csv,sys,uuid,joblib
import numpy as np
import pandas as pd
from collections import Counter
import nltk
#nltk.download("all")
from nltk.stem import WordNetLemmatizer
from string import punctuation, printable
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

## imports from process model scripts
from disaster_tweets_nlp_data_exploration import ingest_data, summarize_data
from disaster_tweets_nlp_data_exploration import DATA_DIR

STOPLIST = ENGLISH_STOP_WORDS
STOPLIST = set(list(STOPLIST) + ["foo"])
SAVED_CORPUS = 'processed-corpus.npz'

def lemmatize_document(doc, stop_words=None):
    """
    Use the WordNetLemmatizer from nltk package
    """
    
    # create an instant of WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    if not stop_words:
        stop_words = set([])
    
    # ensure working with string
    doc = str(doc)
    
    # First remove punctuation from string
    if sys.version_info.major == 3:
        PUNCT_DICT = {ord(punc): None for punc in punctuation}
        doc = doc.translate(PUNCT_DICT)
        
    # remove unicode
    clean_doc = "".join([char for char in doc if char in printable])
    
    tokens = [lemmatizer.lemmatize(w.lower()) for w in clean_doc.split(" ") if len(w)>1]
    
    return " ".join([token for token in tokens if token not in stop_words])


def etl(filename=SAVED_CORPUS, clean=False):
    """
    load, clean, split and save the dataset
    """
    
    saved_corpus = os.path.join(DATA_DIR, filename)
    
    if (not os.path.exists(saved_corpus) or clean):
        
        ## data ingestion
        tweets = ingest_data()
    
        ## only the text and the target will be used from the dataset
        corpus = tweets.text.values
        target = tweets.target.values
        
        ## lemmatize
        print("etl")
        time_start = time.time()
        processed_corpus = [lemmatize_document(tweet, STOPLIST) for tweet in corpus]
        
        ## split the dataset into training and validation set
        train_data, valid_data, y_train, y_valid = train_test_split(processed_corpus, target,
                                                                    test_size=0.25, stratify=target, random_state=42)
        print("---------------------------")
        print("training", sorted(Counter(y_train).items()))
        print("validation", sorted(Counter(y_valid).items()))
        print("---------------------------")
        
        args = {'train_data':train_data,"y_train":y_train,"valid_data":valid_data,"y_valid":y_valid}
        np.savez_compressed(saved_corpus,**args)
        print("process time", time.strftime('%H:%M:%S', time.gmtime(time.time()-time_start)))
        
    else:
        print("loading {} from file".format(saved_corpus))
        npz = np.load(saved_corpus)
        train_data, y_train = npz['train_data'], npz["y_train"]
        valid_data, y_valid = npz['valid_data'], npz["y_valid"]
    
    return train_data, y_train, valid_data, y_valid

if __name__ == "__main__":
    
    run_start = time.time()
    
    ## extract, transform load
    train_data, y_train, valid_data, y_valid = etl(clean=True)
    
    ## summarize data
    summarize_data(np.concatenate((train_data,valid_data),axis=0), preprocessing=False)
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
