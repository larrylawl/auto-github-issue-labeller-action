#!/usr/bin/env python.

import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn
import matplotlib.pyplot as plt

load_dotenv()
ROOT = os.environ.get("ROOT")
# Seen repos
TEXT_EMBEDDINGS_SEEN = f"{ROOT}/pipeline/pickles/text_embeddings_seen.pkl"
TITLE_EMBEDDINGS_SEEN = f"{ROOT}/pipeline/pickles/title_embeddings_seen.pkl"
WORD_COUNT_VECTORS_SEEN = f"{ROOT}/pipeline/pickles/word_count_vectors_seen.pkl"
# HANDCRAFTED_FEATURES_FILE = f"{ROOT}/pipeline/pickles/handcrafted_features.pkl"
TITLE_SENTENCE_EMBEDDINGS_SEEN= f"{ROOT}/pipeline/pickles/title_sentence_embeddings_seen.pkl"
BODY_SENTENCE_EMBEDDINGS_SEEN= f"{ROOT}/pipeline/pickles/body_sentence_embeddings_seen.pkl"

# Seen repos
TEXT_EMBEDDINGS_UNSEEN = f"{ROOT}/pipeline/pickles/text_embeddings_unseen.pkl"
TITLE_EMBEDDINGS_UNSEEN = f"{ROOT}/pipeline/pickles/title_embeddings_unseen.pkl"
WORD_COUNT_VECTORS_UNSEEN = f"{ROOT}/pipeline/pickles/word_count_vectors_unseen.pkl"
TITLE_SENTENCE_EMBEDDINGS_UNSEEN= f"{ROOT}/pipeline/pickles/title_sentence_embeddings_unseen.pkl"
BODY_SENTENCE_EMBEDDINGS_UNSEEN= f"{ROOT}/pipeline/pickles/body_sentence_embeddings_unseen.pkl"

# Thresholds for making a classification decision
THRESHOLDS = [0.33, 0.4, 0.5]

def load_pickle(filename):
    retrieved_df = pd.read_pickle(filename)

    # Manually converting Feature array to suitable format due to some errors at model.fit()
    raw_X = retrieved_df['Feature']
    processed_X = []
    for row in raw_X:
        feature_vector = []
        for value in row:
            feature_vector.append(float(value))
        processed_X.append(feature_vector)

    return processed_X, retrieved_df['Label']

# pass in an array of filepaths as strings
def combine_pickles(files):
    X_dataframes = []
    df_y = [] # df_y is the same for all files
    for filename in files:
        df_X, df_y = load_pickle(filename)
        X_dataframes.append(df_X)

    new_X = []
    for i in range(len(df_y)):
        feature_vector = []
        for df in X_dataframes:
            feature_vector += df[i]
        new_X.append(feature_vector)

    return new_X, df_y

def plot_confusion_matrix(y_true, y_pred):
    confusion_array = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(confusion_array, index = range(3), columns = range(3))
    # plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap='viridis_r', fmt='d')
    plt.show()

# Only class with a probability above threshold will get a classification label
def predict_with_threshold(model, X_test, threshold):
    predictions = model.predict_proba(X_test)
    results = []
    for row in predictions:
        if np.amax(row) >= threshold:
            results.append(np.argmax(row))
        else:
            results.append(-1) # -1 indicates no decision
    
    return results

# Evaluates the model with various metrics at various thresholds of confidence, where `thresholds` is a list of floats from 0 to 1
def evaluate(model, X_test, thresholds, y_test):
    for threshold in thresholds:
        adjusted_pred_seen = predict_with_threshold(model, X_test, threshold=threshold)
        adjusted_accuracy = accuracy_labelled(adjusted_pred_seen, y_test) # only consider accuracy of those labelled
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, adjusted_pred_seen, average="weighted", zero_division=0)  # weighted to account for label imbalance
        results = {
            'threshold': threshold,
            'accuracy': adjusted_accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore
        }
        print(results)

def accuracy_labelled(pred, y_test):
    # fix type issue
    y_test = y_test.to_numpy(dtype=int)

    if len(pred) != len(y_test):
        print("Wrong dimensions!")
        return
    
    total_labelled_instances = 0
    total_correct_instances = 0
    num_doc = 0
    for i in range(len(pred)):
        if pred[i] != -1:
            total_labelled_instances += 1
            if pred[i] == y_test[i]:
                total_correct_instances += 1
            if pred[i] == 2:
                num_doc += 1
    
    # The following stats are just FYI and for checking purposes
    print("total is ", total_labelled_instances)
    print("total correct is ", total_correct_instances)
    print("number of doc issues: ", num_doc)

    return total_correct_instances / total_labelled_instances

def main():
    # Load training data. NOTE: only pass in seen repos here
    print("Combining pickles...")
    df_X_seen, df_y_seen = combine_pickles([TEXT_EMBEDDINGS_SEEN, WORD_COUNT_VECTORS_SEEN])

    # Train-Test split
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(df_X_seen) * 0.8)
    X_train = df_X_seen[:training_length]
    y_train = df_y_seen[:training_length]
    X_test_seen = df_X_seen[training_length:]
    y_test_seen = df_y_seen[training_length:]

    # Training
    model = LogisticRegression(C=1.3, max_iter=2000, multi_class='multinomial')
    print("Training model now...")
    print("X_train length: ", len(X_train))
    # print("10 examples of X_train: ", X_train[:10])
    print("X_train feature length: ", len(X_train[0]))
    print("y_train length: ", len(y_train))
    y_train = y_train.astype('int')
    model.fit(X_train, y_train)

    # Testing for seen repos
    print("Testing model on testing set of SEEN repos now...")
    y_pred_seen = model.predict(X_test_seen)
    y_test_seen = y_test_seen.astype('int')
    score_seen = accuracy_score(y_test_seen, y_pred_seen)
    print('Accurracy score on test set for seen repos = {}'.format(score_seen))
    # plot_confusion_matrix(y_test_seen, y_pred_seen)

    # Evaluating model on seen repos
    print("Evaluating model on SEEN repos:")
    evaluate(model, X_test_seen, THRESHOLDS, y_test_seen)

    # sanity checks
    y_pred_seen = None
    X_test_seen = None
    y_test_seen = None
    score_seen = None

    # Testing for unseen repos
    # load unseen repos first
    df_X_unseen, df_y_unseen = combine_pickles([TEXT_EMBEDDINGS_UNSEEN, WORD_COUNT_VECTORS_UNSEEN])
    print("length of df_X_unseen: ", len(df_X_unseen)) # sanity check
    print("Testing model on UNSEEN repos now...")
    y_pred_unseen = model.predict(df_X_unseen)
    y_test_unseen = df_y_unseen.astype('int')
    score_unseen = accuracy_score(y_test_unseen, y_pred_unseen)
    print('Accurracy score on entire unseen repos = {}'.format(score_unseen))
    # plot_confusion_matrix(y_test_unseen, y_pred_unseen)

    # Evaluating model on unseen repos
    print("Evaluating model on UNSEEN repos:")
    evaluate(model, df_X_unseen, THRESHOLDS, df_y_unseen)

if __name__ == "__main__":
    main()
