#!/usr/bin/env python

import time,os,re,csv,sys,uuid,joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.base import clone
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from disaster_tweets_nlp_etl import etl
from disaster_tweets_nlp_data_exploration import save_fig
from disaster_tweets_nlp_feature_engineering import MODEL_DIR
from disaster_tweets_nlp_feature_engineering import SAVED_MODEL as SAVED_PIPE


RS = 42
SKLEARN_MODEL = "my_sklearn_clf.joblib"
DNN_MODEL = "my_keras_dnn.h5"
EMB_MODEL = "my_keras_emb.h5"

TFHUB_CACHE_DIR = os.path.join(os.curdir, "my_tfhub_cache")
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR
   

def train_ml(train_data, valid_data, y_train, y_valid, model_path=MODEL_DIR, pipe_name=SAVED_PIPE, 
             model_name=SKLEARN_MODEL, rs=RS):
    """
    train classifiers from sklearn
    """
    
    print("...training sklearn classifiers")
    
    ## load transformation pipeline
    feature_engineer = joblib.load(os.path.join(model_path, pipe_name))
    
    ## compare different classifiers
    rd = RidgeClassifier()
    sg = SGDClassifier(tol=1e-3, max_iter=1000)
    rf = RandomForestClassifier(random_state=rs, min_samples_split=250)
    gb = GradientBoostingClassifier(random_state=rs)

    models, scores = {}, {}
    for name, clf in zip(["rd","rf","sg","gb"],[rd,rf,sg,gb]):
        pipe = clone(feature_engineer)
        pipe.steps.append(("clf",clf))
        models[name] = pipe
        scores[name] = cross_val_score(pipe, train_data, y_train, cv=5, scoring="f1")
        
    
    ## plot cross validation scores
    plt.figure(figsize=(10, 5))
    plt.plot([1]*5, scores["rd"], ".")
    plt.plot([2]*5, scores["rf"], ".")
    plt.plot([3]*5, scores["sg"], ".")
    plt.plot([4]*5, scores["gb"], ".")
    plt.boxplot([scores["rd"],scores["rf"],scores["sg"],scores["gb"]],
                labels=("RidgeClassifier","RFClassifier", "SGDClassifier","GBClassifier"))
    plt.ylabel("f1-scores", fontsize=14)
    save_fig("cross_validation_scores")
    
    ## Tune hyper-parameters
    time_start = time.time()
    param_grid = {
        'counter__ngram_range':[(1,1),(1,2),(2,2)],
        'counter__max_features':[1000,5000,10000],
        'clf__n_estimators':[50,70,100],
    }

    grid = GridSearchCV(models["rf"], param_grid=param_grid, cv=3, n_jobs=-1)
    grid.fit(train_data, y_train)

    ## save model
    saved_model = os.path.join(model_path,model_name)
    joblib.dump(grid, saved_model)
    print(grid.best_params_)
    print("train time", time.strftime('%H:%M:%S', time.gmtime(time.time()-time_start)))
    
    
def train_deep(train_data, valid_data, y_train, y_valid, model_path=MODEL_DIR, pipe_name=SAVED_PIPE, 
             model_name=DNN_MODEL, rs=RS):
    """
    train deep learning classifiers
    """
    
    print("...training dense model")
    
    keras.backend.clear_session()
    np.random.seed(rs)
    tf.random.set_seed(rs)
    
    ## load transformation pipeline
    feature_engineer = joblib.load(os.path.join(model_path, pipe_name))
    
    ## transform the datasets
    X_train = feature_engineer.transform(train_data).todense()
    X_valid = feature_engineer.transform(valid_data).todense()
    
    ## build keras dense model
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train.shape[1]))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.AUC()])
    model.summary()
    
    ## training
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid),
                        callbacks=[keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])

    ## save dense model
    saved_model = os.path.join(model_path,model_name)
    model.save(saved_model)
    
    ## evaluate on the validation set
    model.evaluate(X_valid,y_valid)
    
    
def train_embed(train_data, valid_data, y_train, y_valid, model_path=MODEL_DIR, model_name=EMB_MODEL, rs=RS):
    """
    train deep learning classifier reusing pretrained embeddings
    """
    
    print("...training dense model reusing pretrained embeddings")
    
    keras.backend.clear_session()
    np.random.seed(rs)
    tf.random.set_seed(rs)
    
    ## build keras model
    model = keras.models.Sequential()
    model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                             dtype=tf.string, input_shape=[], output_shape=[50], trainable=True))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.AUC()])
    model.summary()
    
    ## training
    history = model.fit(train_data, y_train, epochs=10,validation_data=(valid_data, y_valid),
                        callbacks=[keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])

    ## save model
    saved_model = os.path.join(model_path,model_name)
    model.save(saved_model)
    
    ## evaluate on the validation set
    model.evaluate(valid_data,y_valid)
    
    
def plot_roc_curve(fpr, tpr, label=None):
    """
    plot ROC curves
    """
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--', label='Dummy')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    
def model_evaluation(input_data, target, set_name, model_path=MODEL_DIR, sklearn_model=SKLEARN_MODEL, 
                     dnn_model=DNN_MODEL, emb_model=EMB_MODEL, pipe_name=SAVED_PIPE):
    """
    evaluate the three models
    """
    
    print("------------------------------")
    print("Model Evaluation on the {} Set".format(set_name.upper()))
    
    ## load trained models
    clf = joblib.load(os.path.join(model_path,sklearn_model))
    dnn = keras.models.load_model(os.path.join(model_path,dnn_model))
    emb = keras.models.load_model(os.path.join(model_path,emb_model),custom_objects={'KerasLayer':hub.KerasLayer})
    
    ## load transformation pipeline
    feature_engineer = joblib.load(os.path.join(model_path, pipe_name))
    
    ## transform the input dataset (needed for the dnn model)
    X = feature_engineer.transform(input_data).todense()
    
    ## calculate scores
    clf_scores = clf.predict_proba(input_data)[:,1]
    #clf_scores = clf.predict(input_data).ravel()
    dnn_scores = dnn.predict(X).ravel()
    emb_scores = emb.predict(input_data).ravel()
    
    ## calculate predictions
    clf_pred = clf.predict(input_data)
    dnn_pred = dnn.predict_classes(X)
    emb_pred = emb.predict_classes(input_data)
    
    ## calculate metrics
    clf_fpr, clf_tpr, clf_thresholds = roc_curve(target, clf_scores)
    dnn_fpr, dnn_tpr, dnn_thresholds = roc_curve(target, dnn_scores)
    emb_fpr, emb_tpr, emb_thresholds = roc_curve(target, emb_scores)
    
    ## plot roc curves
    plt.figure(figsize=(8, 6))
    plt.plot(clf_fpr, clf_tpr, 'r--', label='RNF')
    plt.plot(dnn_fpr, dnn_tpr, 'g--', label='DNN')
    plot_roc_curve(emb_fpr, emb_tpr, "EMB")
    plt.legend(loc='lower right')
    save_fig("plot_curves_{}_set".format(set_name))
    
    ## print classification report
    for name, scores, y_pred in zip(["Random Forest","DNN","EmbedDNN"],
                                    [clf_scores,dnn_scores,emb_scores],
                                    [clf_pred,dnn_pred,emb_pred]):
        print(name.upper())
        print(classification_report(target, y_pred))
        print('ROC AUC:', roc_auc_score(target, scores))
        print('f1-score:{}'.format(f1_score(target, y_pred)))
    
    
def model_train(train_data, y_train, rs=RS):
    """
    train classifiers and select the best model
    """
    
    print("Model Training")
    
    ## create validation set from training set
    train_data, valid_data, y_train, y_valid = train_test_split(train_data, y_train, test_size=0.2,
                                                                stratify=y_train, random_state=rs)
    
    print("train", sorted(Counter(y_train).items()))
    print("valid", sorted(Counter(y_valid).items()))
    print("test", sorted(Counter(y_test).items()))
    
    ## train machine learning classifier
    train_ml(train_data, valid_data, y_train, y_valid)
    
    ## train deep learning classifier
    train_deep(train_data, valid_data, y_train, y_valid)
    
    ## train deep classifier with pretrained embeddings
    train_embed(train_data, valid_data, y_train, y_valid)
    
    ## model evaluation on train set
    model_evaluation(train_data, y_train, set_name="train")  
    
    ## model evaluation on validation set
    model_evaluation(valid_data, y_valid, set_name="valid")  
    
    
if __name__ == "__main__":
    
    run_start = time.time()
    
    ## load the dataset
    train_data, y_train, test_data, y_test = etl()
    
    ## model training
    model_train(train_data, y_train)
    
    ## model evaluation on test set
    model_evaluation(test_data, y_test, set_name="test")
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
    
