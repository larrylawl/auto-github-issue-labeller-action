#!/usr/bin/env python.

import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
ROOT = os.environ.get("ROOT")
FEATURE = "body"  # [title, body]
DEVICE = "cuda"  # "cpu/cuda"
MODEL = 'distilbert-base-nli-stsb-mean-tokens'
LOAD_TRAIN_PATH = f"{ROOT}/pipeline/pickles/dataframe_train.pkl"
LOAD_TEST_PATH = f"{ROOT}/pipeline/pickles/dataframe_test.pkl"

###### This script generates /pickles/sentence_embedding.pkl ######
# Average of sentence embeddings

def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text


def load_dataframe_from_pickle(path):
    retrieved_df = pd.read_pickle(path)
    return retrieved_df


def avg_sentence_embedding(paragraph, model):
    """ Returns average of the sentence embedding vectors in the paragraph. """
    all_embeddings = model.encode(nltk.sent_tokenize(paragraph))
    avg_embedding_np = np.nanmean(all_embeddings, axis=0) if not np.isnan(all_embeddings).all() else np.zeros(768)
    return avg_embedding_np

def save_vector_array(vector_array, labels, filename):
    save_df = pd.DataFrame(columns=['Feature', 'Label'])
    save_df['Feature'] = pd.Series(vector_array)
    save_df['Label'] = labels.values
    save_df.to_pickle(filename)

def main():
    train_df = load_dataframe_from_pickle(LOAD_TRAIN_PATH)
    test_df = load_dataframe_from_pickle(LOAD_TEST_PATH)
    print("Done loading dataframe.")

    # Removing Markdown
    train_df[FEATURE] = train_df[FEATURE].apply(lambda x: remove_markdown(x))
    test_df[FEATURE] = test_df[FEATURE].apply(lambda x: remove_markdown(x))
    print("Done with removing Markdown.")

    model = SentenceTransformer(MODEL, device=DEVICE)
    train_sent_embeddings= []  # 1-D
    test_sent_embeddings = []  # 1-D
    if FEATURE == "title":
        for _, sent in train_df["title"].items():
            train_sent_embeddings.append(model.encode(sent))
        for _, sent in test_df["title"].items():
            test_sent_embeddings.append(model.encode(sent))
    elif FEATURE == "body":
        for _, para in train_df["body"].items():
            train_sent_embeddings.append(avg_sentence_embedding(para, model))
        for _, para in test_df["body"].items():
            test_sent_embeddings.append(avg_sentence_embedding(para, model))
    else:
        raise NotImplementedError("Only supports embedding of title and text for now.")
    print("Done with sentence embeddings.")

    print("Saving feature vectors to disc...")
    train_filename = f"{ROOT}/pipeline/pickles/{FEATURE}_sentence_embeddings_train.pkl"
    save_vector_array(train_sent_embeddings, train_df['labels'], filename=train_filename)

    test_filename = f"{ROOT}/pipeline/pickles/{FEATURE}_sentence_embeddings_test.pkl"
    save_vector_array(test_sent_embeddings, test_df['labels'], filename=test_filename)
    print("Done with saving.")


if __name__ == "__main__":
    main()
