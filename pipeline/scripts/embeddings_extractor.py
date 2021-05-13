#!/usr/bin/env python.

import os
import math
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from dotenv import load_dotenv


###### This script generates the following pickle files: ######
# /pickles/text_embeddings_seen.pkl
# /pickles/title_embeddings_seen.pkl
# /pickles/text_embeddings_unseen.pkl
# /pickles/title_embeddings_unseen.pkl

load_dotenv()
ROOT = os.environ.get("ROOT")

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2
}

# Given an array of sentences, transform each sentence into a vector
# representing the average of normalised embeddings of all the words.
def generate_averaged_word_embeddings(embeddings_model, sentences):
    def normalise(x):
        sum_of_squares = np.sum(x**2)
        l2_norm = np.sqrt(sum_of_squares)
        if l2_norm > 0:
            return (1.0 / l2_norm) * x
        else:
            return x
    
    res = []
    num_of_sentences = 0
    for sentence in sentences:
        num_of_sentences += 1 

        pattern = r"\w+'\w+|\w+-\w+|\w+|[(...).,!:\";\?]"
        tokens = re.findall(pattern, sentence)
        summed_vector = None
        is_first = True
        length = 0

        for token in tokens:
            if token in embeddings_model:
                embedding = normalise(embeddings_model[token])
                length += 1
                if is_first:
                    summed_vector = embedding
                    is_first = False
                else:
                    summed_vector = summed_vector + embedding
        if summed_vector is None:
            summed_vector = np.zeros(300)
            length = 1
        averaged_vector = summed_vector / length
        res.append(averaged_vector)

        if num_of_sentences % 1000 == 0:
            print("Done with vectorising # of sentences: ", num_of_sentences)

    return res

# Given a String of sentence, returns an array of tokens
def tokenise(sentence):
    pattern = r'\w+'
    return re.findall(pattern, sentence)

# sg=1 for skip-gram; sg=0 for CBOW
def train_embeddings(df, size, window, sg):
    sentences = []
    for _, row in df.iterrows():
        sentences.append(tokenise(row['body']))
        sentences.append(tokenise(row['title']))

    embeddings_model = Word2Vec(sentences=sentences, size=size, window=window, sg=sg)
    return embeddings_model

def load_dataframe_from_pickle(is_seen):
    if is_seen:
        return pd.read_pickle(f"{ROOT}/pipeline/pickles/dataframe_train.pkl")
    else:
        return pd.read_pickle(f"{ROOT}/pipeline/pickles/dataframe_test.pkl")

def save_vector_array(vector_array, labels, filename):
    save_df = pd.DataFrame(columns=['Feature', 'Label'])
    save_df['Feature'] = pd.Series(vector_array)
    save_df['Label'] = labels.values
    save_df.to_pickle(filename)

def main():
    # seen repos
    seen_df = load_dataframe_from_pickle(is_seen=True)
    print("Done loading dataframe_train.pkl.")

    embeddings_model = train_embeddings(seen_df, size=300, window=5, sg=0) # 0 for CBOW
    print("Done with embeddings training.")

    text_vector_array_seen = generate_averaged_word_embeddings(embeddings_model, seen_df['body'])
    print("Done with text embedding vectorisation.")
    title_vector_array_seen = generate_averaged_word_embeddings(embeddings_model, seen_df['title'])
    print("Done with title embedding vectorisation.")

    save_vector_array(text_vector_array_seen, seen_df['labels'], filename=f"{ROOT}/pipeline/pickles/text_embeddings_seen.pkl")
    print("Done with saving text vector array")

    save_vector_array(title_vector_array_seen, seen_df['labels'], filename=f"{ROOT}/pipeline/pickles/title_embeddings_seen.pkl")
    print("Done with saving title vector array")

    # unseen repos
    unseen_df = load_dataframe_from_pickle(is_seen=False)
    print("Done loading dataframe_test.pkl.")

    text_vector_array_unseen = generate_averaged_word_embeddings(embeddings_model, unseen_df['body'])
    print("Done with text embedding vectorisation.")
    title_vector_array_unseen = generate_averaged_word_embeddings(embeddings_model, unseen_df['title'])
    print("Done with title embedding vectorisation.")

    save_vector_array(text_vector_array_unseen, unseen_df['labels'], filename=f"{ROOT}/pipeline/pickles/text_embeddings_unseen.pkl")
    print("Done with saving text vector array")

    save_vector_array(title_vector_array_unseen, unseen_df['labels'], filename=f"{ROOT}/pipeline/pickles/title_embeddings_unseen.pkl")
    print("Done with saving title vector array")

if __name__ == "__main__":
    main()
