#!/usr/bin/env python.
import os

import pandas as pd
from dotenv import load_dotenv

###### This script generates /pickles/dataframe_train.pkl and /pickles/dataframe_test.pkl ######

load_dotenv()
ROOT = os.environ.get("ROOT")

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2,
    "others": -1,
}

TRAIN_MAPPINGS = {
    "tensorflow": {
        "feature": ["type:feature"],
        "bug": ["type:bug"],
        "doc": ["type:docs-feature", "type:docs-bug"],
    },
    "rust": {
        "feature": ["C-feature-request", "C-feature-accepted", "C-enhancement"],
        "bug": ["C-bug"],
        "doc": ["T-doc"],
    },
    "kubernetes": {
        "feature": ["kind/feature", "kind/api-change"],
        "bug": ["kind/bug"],
        "doc": ["kind/documentation"],
    }
}

TEST_MAPPINGS = {
    "flutter": {
        "feature": ['severe: new feature'],
        "bug": ["severe: crash", "severe: fatal crash", "severe: rendering"],
        "doc": ["documentation"],
    },
    "ohmyzsh": {
        "feature": ["Feature", "Enhancement"],
        "bug": ["Bug"],
        "doc": ["Type: documentation"],
    },
    "electron": {
        "feature": ["enhancement :sparkles:"],
        "bug": ["bug :beetle:", "crash :boom:"],
        "doc": ["documentation :notebook:"],
    }
}


def standardise_df_labels(df, feature_labels, bug_labels, doc_labels):
    def standardise(label):
        if label in feature_labels:
            return LABELS['feature']
        elif label in bug_labels:
            return LABELS['bug']
        elif label in doc_labels:
            return LABELS['doc']
        else:
            return LABELS['others']

    # df['class'] = df['labels'].apply(standardise)  # sanity check
    df['labels'] = df['labels'].apply(standardise)
    return df

def remove_redundant_classes(df):
    is_correct_class = df['labels'] != -1
    df = df[is_correct_class]
    return df

def main():
    def _generate_dataframe(mapping, file_path):
        dfs = []

        for repo_name, repo_details in mapping.items():
            # load data
            df = pd.read_json(f'{ROOT}/data/eng_labelled/remove_duplicates/{repo_name}_removed_dups.json')

            # standardise labels
            df = standardise_df_labels(df, repo_details["feature"], repo_details["bug"], repo_details["doc"])

            # remove other class
            df = remove_redundant_classes(df)

            # append to df
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=1)  # seed randomisation
        print(combined_df.shape)  # sanity check
        combined_df.to_pickle(file_path)
    print("Generating dfs...")
    _generate_dataframe(TRAIN_MAPPINGS, f"{ROOT}/pipeline/pickles/dataframe_train.pkl")
    _generate_dataframe(TEST_MAPPINGS, f"{ROOT}/pipeline/pickles/dataframe_test.pkl")


if __name__ == "__main__":
    main()
