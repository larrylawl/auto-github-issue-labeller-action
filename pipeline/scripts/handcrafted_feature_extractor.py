#!/usr/bin/env python.

import re
import pandas as pd

###### This script generates /pickles/handcrafted_features.pkl ######

def tokenise(string):
    pattern = r'\w+'
    return re.findall(pattern, string)

def distinct_word_count(string):
    tokens = tokenise(string)
    for token in tokens:
        token = token.lower()
    return len(set(tokens))

def uppercase_word_count(string):
    tokens = tokenise(string)
    count = 0
    for token in tokens:
        if token[0].isupper():
            count += 1
    return count

def github_username_count(string):
    pattern = r'\s@.+\s'
    tokens = re.findall(pattern, string)
    return len(tokens)

def issue_reference_count(string):
    pattern = r'\s#[\d]+\s'
    tokens = re.findall(pattern, string)
    return len(tokens)

def link_count(string):
    pattern = r'\s\[(.*?)\]\((.*?)\)\s'
    tokens = re.findall(pattern, string)
    return len(tokens)

def image_count(string):
    pattern = r'\s!\[[^\]]*\]\((.*?)\s*("(?:.*[^"])")?\s*\)\s'
    tokens = re.findall(pattern, string)
    return len(tokens)

def code_count(string):
    pattern = r'``|```'
    tokens = re.findall(pattern, string)
    return len(tokens)

def asterisk_count(string): 
    pattern = r'\*'
    tokens = re.findall(pattern, string)
    return len(tokens)

def underscore_count(string):
    pattern = r'_'
    tokens = re.findall(pattern, string)
    return len(tokens)

def hash_count(string):
    pattern = r'#'
    tokens = re.findall(pattern, string)
    return len(tokens)

# For one data point
# NOTE: Can refine later
def generate_handcrafted_feature_vector(data):
    feature_vector = []

    #Feature 1: length of title
    feature_vector.append(len(data['title']))

    #Feature 2: length of description
    feature_vector.append(len(data['text']))

    #Feature 3: length of codeblocks
    feature_vector.append(len(data['codeblocks']))

    #Feature 4: distinct word count
    feature_vector.append(distinct_word_count(data['text']))

    #Feature 5: uppercase word count
    feature_vector.append(uppercase_word_count(data['text']))

    #Feature 6: GitHub username count
    feature_vector.append(github_username_count(data['text']))

    #Feature 7: GitHub issue reference count
    feature_vector.append(issue_reference_count(data['text']))

    #Feature 8: link count
    feature_vector.append(link_count(data['text']))

    #Feature 9: image count
    feature_vector.append(image_count(data['text']))

    #Feature 10: code (``` or ``) count
    feature_vector.append(code_count(data['text']))

    #Feature 11: asterisk(bold) count
    feature_vector.append(asterisk_count(data['text']))

    #Feature 12: underscore(italics) count
    feature_vector.append(underscore_count(data['text']))

    #Feature 13: hash(section) count
    feature_vector.append(hash_count(data['text']))

    return feature_vector

# For all data points
def generate_feature_matrix(df):
    X_matrix = []
    for _, row in df.iterrows():
        feature_vector = generate_handcrafted_feature_vector(row)
        X_matrix.append(feature_vector)
    
    return X_matrix

def load_dataframe_from_pickle():
    retrieved_df = pd.read_pickle("../pickles/dataframe.pkl")
    return retrieved_df

def save_vector_array(vector_array, labels, filename):
    save_df = pd.DataFrame(columns=['Feature', 'Label'])
    save_df['Feature'] = pd.Series(vector_array)
    save_df['Label'] = labels.values
    save_df.to_pickle(filename)

def main():
    df = load_dataframe_from_pickle()
    print("Done loading dataframe.")

    feature_vectors = generate_feature_matrix(df)
    print("Done with generating feature vectors.")

    print("Saving feature vectors to memory...")      
    save_vector_array(feature_vectors, df['labels'], filename="../pickles/handcrafted_features.pkl")
    print("Done with saving.")

if __name__ == "__main__":
    main()
