#!/usr/bin/env python.

import os
import pickle
import re

import pandas as pd
from dotenv import load_dotenv

###### This script generates /pickles/sequence_features.pkl ######
# Format of the pickle is an array of arrays of length 3 [title, text, label]

load_dotenv()
ROOT = os.environ.get("ROOT")
LOAD_TRAIN_PATH = f"{ROOT}/pipeline/pickles/dataframe_train.pkl"
SAVE_TRAIN_PATH = f"{ROOT}/pipeline/pickles/sequence_features_train.pkl"
LOAD_TEST_PATH = f"{ROOT}/pipeline/pickles/dataframe_test.pkl"
SAVE_TEST_PATH = f"{ROOT}/pipeline/pickles/sequence_features_test.pkl"


def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text


def load_dataframe_from_pickle(path):
    retrieved_df = pd.read_pickle(path)
    return retrieved_df


def main():
    def _generate_seq_features(load_path, save_path):
        df = load_dataframe_from_pickle(load_path)
        print("Done loading dataframe.")

        # Removing Markdown
        results = []
        for _, row in df.iterrows():
            results.append([remove_markdown(row['title']), remove_markdown(row['body']), row['labels']])
        print("Done with removing Markdown.")
        print(len(results[0]))

        print("Saving to pickle...")
        outfile = open(save_path, 'wb')
        pickle.dump(results, outfile)
        outfile.close()
        print("Done with pickling.")

    _generate_seq_features(LOAD_TRAIN_PATH, SAVE_TRAIN_PATH)
    _generate_seq_features(LOAD_TEST_PATH, SAVE_TEST_PATH)


if __name__ == "__main__":
    main()
