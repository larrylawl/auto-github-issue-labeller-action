# Auto-Labelling of GitHub Issues using Natural Language Processing 
Code accompanying our paper [here](TODO).

## Installation
We use python version 3.6.5. 

1. We need to first install the dependencies. 

```
pip install -r requirements.txt
```

2. Specify environment variables in `{ROOT}/.env` file. A sample schema is provided in `{ROOT}/.env-sample`.

3. **Optional**  Download the directory containing our final model in the root directory at this [link](https://drive.google.com/file/d/1JO_I8GNDDwLY4hIeySdhIMwI23kM7Ijx/view?usp=sharing]).

## Scraping data
We need to first scrape the issues from open-sourced repos. 

```
python src/data_collection/scraper.py src/data_collection/english_schema/tensorflow_schema.json
```

## Data preprocessing
To generate dataframes which are used as inputs for both our neural network and BERT models, run

```
python pipeline/scripts/dataframe_generator.py   
```

This script generates `{ROOT}/pickles/dataframe_train.pkl` and `{ROOT}/pickles/dataframe_test.pkl`.

## Running of models
To run our BERT classifier,

```
python pipeline/scripts/seq_classifier.py
```

You can further modify the options within the file.

