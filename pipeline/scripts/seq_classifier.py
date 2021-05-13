#!/usr/bin/env python.
import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

from utils import remove_log, remove_code_block, remove_url, remove_markdown, has_log, has_code_block, \
    has_url, average_results, accuracy_labelled

load_dotenv()
ROOT = os.environ.get("ROOT")

options = {
    "preprocess": [remove_url, remove_log],  # remove_code_block, remove_url, remove_log
    "features": ["title", "body"],  # title, body
    "load_train_path": f"{ROOT}/pipeline/pickles/dataframe_train.pkl",
    "load_test_path": f"{ROOT}/pipeline/pickles/dataframe_test.pkl",
    "save_dir": f"{ROOT}/results/temp",
    # "load_dir": f"{ROOT}/results/final-log",  # If None, will train from scratch,
    "load_dir": None,
    "n_repeat": 3,
    "test_mode": True,
    "confidence": -10,  # [-10, 2, 4]. Threshold for logit output. -10 is ~= argmax.
    "device": torch.device("cuda"),  # cpu, cuda
    "train_test_split": 0.8,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_steps": 10
}

# Consts
ALL_TEST_DS_TYPE = ["seen", "unseen"]
DS_TYPE = ["all", "code", "url", "log"]



class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess(df):
    for fn in options["preprocess"]:
        df = df.apply(fn)
    return df


def load_dataframe_from_pickle(path):
    retrieved_df = pd.read_pickle(path)
    return retrieved_df


def pretty_dict(dict):
    """ Returns a pretty string version of a dictionary.
    """
    result = ""
    for key, value in dict.items():
        key = str(key)
        value = str(value)
        if len(value) < 40:
            result += f'{key}: {value} \n'
        else:
            result += f'{key}: \n' \
                      f'{value} \n'
    return result


def prep_single_dataset(df, tokenizer):
    df = df.copy(deep=True)
    encodings = tokenizer(preprocess(df["X"]).tolist(), truncation=True, padding=True)
    dataset = Dataset(encodings, df["labels"].tolist())
    return dataset


def prep_datasets(df, tokenizer):
    """ Returns all, code, url, log preprocessed dataset from list of training data (str) and list of labels (int). """
    all_ds = prep_single_dataset(df, tokenizer)
    result = [all_ds]
    fns = [has_code_block, has_url, has_log]
    print(f"Shape before filtering: {df.shape}")
    for fn in fns:
        df_ = df[df["X"].apply(fn)]
        print(f"Shape after {fn}: {df_.shape}")
        assert df_.size <= df.size, "filtered df should not be of bigger size than original one"
        if df_.size > 0:
            ds = prep_single_dataset(df_, tokenizer)
        else:
            ds = []
            print(f"WARNING: dataset for {fn} empty. Ensure this is intended.")
        result.append(ds)

    return result


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.empty(labels.shape)
    for i, pred in enumerate(pred.predictions):
        preds[i] = np.argmax(pred) if np.amax(pred) > options["confidence"] else -1
    acc = accuracy_labelled(preds, labels)
    # acc = accuracy_score(labels, preds)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average="weighted")  # weighted to account for label imbalance
    # cm = confusion_matrix(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }

def model_init():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    return model

def train_model(train_dataset, save_dir, seed):
    print("Training model...")
    no_cuda = options["device"] == torch.device('cpu')
    training_args = TrainingArguments(
        output_dir=save_dir,  # output directory
        num_train_epochs=options["num_train_epochs"],  # total number of training epochs
        per_device_train_batch_size=options["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=options["per_device_eval_batch_size"],  # batch size for evaluation
        warmup_steps=options["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=options["weight_decay"],  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=options["logging_steps"],
        no_cuda=no_cuda,
        save_strategy="no",
        seed=seed
    )

    trainer = Trainer(
        model=None,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        compute_metrics=compute_metrics,  # accuracy metric
        model_init=model_init  # control random weights in model
    )

    trainer.train()

    return trainer


def load_model(save_dir, load_path):
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(load_path, num_labels=3)
    no_cuda = options["device"] == torch.device('cpu')
    training_args = TrainingArguments(  # no trg is done
        output_dir=save_dir,
        no_cuda=no_cuda,
        logging_dir='./logs',  # directory for storing logs
    )

    trainer = Trainer(
        model=model,  # loading pre trained model
        args=training_args,
        compute_metrics=compute_metrics  # accuracy metric
    )

    return trainer


def main():
    print("Preparing data...")
    # accumulated results: [[[], [], [], []], [[], [], [], []]]
    results = []
    for i in range(len(ALL_TEST_DS_TYPE)):
        results.append([])
        for j in range(len(DS_TYPE)):
            results[i].append([])
    assert len(results) == len(ALL_TEST_DS_TYPE)
    assert len(results[0]) == len(DS_TYPE)

    for i in range(options["n_repeat"]):
        # Setting seeds to control randomness
        seed = i
        np.random.seed(seed)
        torch.manual_seed(seed)

        # create dir
        save_dir_repeat = os.path.join(options["save_dir"], f"repeat_{i}")
        if not os.path.exists(save_dir_repeat):
            os.makedirs(save_dir_repeat)

        # Load stuff
        if options["load_dir"]:
            load_path = os.path.join(options["load_dir"], f"repeat_{i}")
            tokenizer = DistilBertTokenizerFast.from_pretrained(load_path)
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        train_data = load_dataframe_from_pickle(options["load_train_path"])
        test_data = load_dataframe_from_pickle(options["load_test_path"])

        # Retrieve features
        print("Retrieving features...")
        train_data['X'] = ''
        test_data['X'] = ''
        for feature in options["features"]:
            train_data['X'] += train_data[feature] + " "
            test_data['X'] += test_data[feature] + " "

        # Preprocess
        # print("Preprocessing...")
        if options["test_mode"]:
            train_data = train_data[:50]
            test_data = test_data[:50]

        # Preparing model
        print("Preparing model...")
        # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
        training_length = math.ceil(len(train_data.index) * options["train_test_split"])
        if not bool(options["load_dir"]): train_dataset = prep_single_dataset(train_data[:training_length], tokenizer)
        test_seen_datasets = prep_datasets(train_data[training_length:], tokenizer)  # all, code, url, log
        test_unseen_datasets = prep_datasets(test_data, tokenizer)
        assert len(test_seen_datasets) == 4
        assert len(test_unseen_datasets) == 4
        del train_data
        del test_data

        # Building model
        if bool(options["load_dir"]):
            trainer = load_model(save_dir_repeat, load_path)
        else:  # train from scratch
            trainer = train_model(train_dataset, save_dir_repeat, seed)
            print("Saving the good stuff in case they get lost...")
            trainer.save_model(save_dir_repeat)
            tokenizer.save_pretrained(save_dir_repeat)

        print("Evaluating...")
        all_test_ds = [test_seen_datasets, test_unseen_datasets]
        for i, test_ds in enumerate(all_test_ds):
            for j, ds in enumerate(test_ds):
                result = trainer.evaluate(ds) if len(ds) > 0 else {}
                result["dataset size"] = len(ds)
                results[i][j].append(result)

    # combining all results for each test dataset
    info = options
    for i, test_ds in enumerate(ALL_TEST_DS_TYPE):
        for j, ds in enumerate(DS_TYPE):
            avg_results = average_results(results[i][j])
            info[f"avg results on {ALL_TEST_DS_TYPE[i]} {DS_TYPE[j]} repos"] = avg_results
            info[f"all results on {ALL_TEST_DS_TYPE[i]} {DS_TYPE[j]} repos"] = results[i][j]

    # saving results and model
    print("Saving all the good stuff...")
    data_file = open(f'{options["save_dir"]}/data.txt', "w+")
    data_file.write(pretty_dict(info))
    data_file.close()


if __name__ == "__main__":
    main()
