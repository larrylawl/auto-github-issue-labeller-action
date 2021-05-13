#!/usr/bin/env python.

import os
from collections import Counter
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

###### This script generates the following pickle files: ######
# /pickles/word_count_vectors_seen.pkl
# /pickles/word_count_vectors_unseen.pkl

load_dotenv()
ROOT = os.environ.get("ROOT")

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2
}

def generate_word_count_features(sentence_matrix, influential_words=[]):
    processed_X = []
    for corpus in sentence_matrix:
        vector = []
        tokens = re.findall(r'\w+', corpus)

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        counter = Counter(tokens)
        for word in influential_words:
            if word.lower() in counter:
                vector.append(counter[word.lower()])
            else:
                vector.append(0)
        
        # print("vector is ", vector)
        processed_X.append(vector)
    
    return processed_X

# Removes all occurrences of words in `target` list from the `words` list.
def remove_words_from(words, target):
    res = []
    for word in words:
        if not (word in target):
            res.append(word)
    return res

# Given a String of sentence, returns an array of tokens
def tokenise(sentence):
    pattern = r'\w+'
    return re.findall(pattern, sentence)

# Returns a set of n most frequent words in each category excluding m common most frequent words
# in the combined corpus.
def top_words_in_categories(n, m, sentences, labels):
    feature_examples = []
    bug_examples = []
    doc_examples = []
    for index, example in enumerate(sentences):
        if labels[index] == LABELS['feature']:
            feature_examples.append(example)
        elif labels[index] == LABELS['bug']:
            bug_examples.append(example)
        elif labels[index] == LABELS['doc']:
            doc_examples.append(example)
        else:
            print("should not reach here")

    counter = Counter(re.findall(r'\w+', " ".join(sentences)))
    most_common = counter.most_common(m)
    most_common = list(map(lambda x: x[0], most_common))
    print("most frequent common words are: ", most_common)

    feature_tokens = re.findall(r'\w+', " ".join(feature_examples))
    feature_tokens_without_common_words = remove_words_from(feature_tokens, most_common)
    counter_feature = Counter(feature_tokens_without_common_words)
    most_common_feature = counter_feature.most_common(n)
    most_common_feature = list(map(lambda x: x[0], most_common_feature))
    print("most frequent FEATURE words are: ", most_common_feature)

    bug_tokens = re.findall(r'\w+', " ".join(bug_examples))
    bug_tokens_without_common_words = remove_words_from(bug_tokens, most_common)
    counter_bug = Counter(bug_tokens_without_common_words)
    most_common_bug = counter_bug.most_common(n)
    most_common_bug = list(map(lambda x: x[0], most_common_bug))
    print("most frequent BUG words are: ", most_common_bug)

    doc_tokens = re.findall(r'\w+', " ".join(doc_examples))
    doc_tokens_without_common_words = remove_words_from(doc_tokens, most_common)
    counter_doc = Counter(doc_tokens_without_common_words)
    most_common_doc = counter_doc.most_common(n)
    most_common_doc = list(map(lambda x: x[0], most_common_doc))
    print("most frequent DOC words are: ", most_common_doc)

    return set(most_common_feature + most_common_bug + most_common_doc)

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

    # generate influential words
    influential_words = list(top_words_in_categories(150, 50, seen_df['body'], seen_df['labels']))
    word_count_vectors_seen = generate_word_count_features(seen_df['body'], influential_words)
    print("Done with generating word count vectors for seen repos.")
     
    save_vector_array(word_count_vectors_seen, seen_df['labels'], filename=f"{ROOT}/pipeline/pickles/word_count_vectors_seen.pkl")
    print("Done with saving.")

    # unseen repos
    unseen_df = load_dataframe_from_pickle(is_seen=False)
    print("Done loading dataframe_test.pkl.")
    
    word_count_vectors_unseen = generate_word_count_features(unseen_df['body'], influential_words)
    print("Done with generating word count vectors for unseen repos.")
     
    save_vector_array(word_count_vectors_unseen, unseen_df['labels'], filename=f"{ROOT}/pipeline/pickles/word_count_vectors_unseen.pkl")
    print("Done with saving.")

if __name__ == "__main__":
    main()
