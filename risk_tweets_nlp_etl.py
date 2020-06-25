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
from risk_tweets_nlp_data_exploration import ingest_data, summarize_data
from risk_tweets_nlp_data_exploration import DATA_DIR

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


def etl(clean=False, filename=SAVED_CORPUS, data_path=DATA_DIR, stop_list=STOPLIST):
    """
    load, clean, split and save the dataset
    """
    
    saved_corpus = os.path.join(data_path, filename)
    
    if (not os.path.exists(saved_corpus) or clean):
        
        ## data ingestion
        tweets = ingest_data()
    
        ## only the text and the target will be used from the dataset
        corpus = tweets.text.values
        target = tweets.target.values
        
        ## lemmatize
        print("ETL")
        time_start = time.time()
        processed_corpus = [lemmatize_document(tweet, stop_list) for tweet in corpus]
        
        ## split the dataset into training and test set
        train_data, test_data, y_train, y_test = train_test_split(processed_corpus, target,
                                                                  test_size=0.2, stratify=target, random_state=42)
        print("---------------------------")
        print("train", sorted(Counter(y_train).items()))
        print("test", sorted(Counter(y_test).items()))
        print("---------------------------")
        
        args = {'train_data':train_data,"y_train":y_train,"test_data":test_data,"y_test":y_test}
        np.savez_compressed(saved_corpus,**args)
        print("process time", time.strftime('%H:%M:%S', time.gmtime(time.time()-time_start)))
        
    else:
        print("loading {} from file".format(saved_corpus))
        npz = np.load(saved_corpus)
        train_data, y_train = npz['train_data'], npz["y_train"]
        test_data, y_test = npz['test_data'], npz["y_test"]
    
    return train_data, y_train, test_data, y_test

if __name__ == "__main__":
    
    run_start = time.time()
    
    ## extract, transform load
    train_data, y_train, test_data, y_test = etl(clean=True)
    
    ## summarize data
    summarize_data(np.concatenate((train_data,test_data),axis=0), preprocessing=False)
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
