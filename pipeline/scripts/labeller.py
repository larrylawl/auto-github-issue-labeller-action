import datetime
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from utils import remove_code_block, remove_url

load_dotenv()  # for local development
DELTA = int(os.environ.get("DELTA"))
SECRET_TOKEN = os.environ.get("GITHUB_TOKEN")
REPOSITORY = os.environ.get("REPOSITORY")
CONFIDENCE = int(os.environ.get("CONFIDENCE"))
FEATURE = os.environ.get("FEATURE")
BUG = os.environ.get("BUG")
DOCS = os.environ.get("DOCS")
LABELS = [FEATURE, BUG, DOCS, None]
FINAL_MODEL_DIR = "./final_model"
TOKENIZER = DistilBertTokenizerFast.from_pretrained(FINAL_MODEL_DIR)
MODEL = DistilBertForSequenceClassification.from_pretrained(FINAL_MODEL_DIR, num_labels=3)
MODEL.eval()


def preprocess(text):
    preprocessed_text = remove_url(remove_code_block(text))
    return preprocessed_text


def classify(title, body):
    ''' Returns one of "bug", "documentation", "feature" '''
    preprocessed_text = preprocess(f"{title} {body}")
    encodings = TOKENIZER(preprocessed_text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    outputs = MODEL(input_ids, attention_mask=attention_mask)
    pred = outputs["logits"].detach().numpy()
    pred = np.argmax(pred) if np.amax(pred) > CONFIDENCE else -1

    return LABELS[pred]


s = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    "Authorization": "token " + SECRET_TOKEN,
    "Accept": "application/vnd.github.v3+json"
}


def add_label(issue_num, label):
    res = s.post(f'https://api.github.com/repos/{REPOSITORY}/issues/{issue_num}/labels', headers=headers,
                json={"labels": [label]})


def main():
    print("Starting labeller...")

    ## For some reasons, when set to now(), the REST API returns nothing.
    ## So we add some interval
    payload = {
        "since": (datetime.datetime.now() - datetime.timedelta(days=DELTA)).isoformat()
    }

    res = s.get(f'https://api.github.com/repos/{REPOSITORY}/issues', headers=headers, params=payload)
    time.sleep(3)

    res = res.json()
    if len(res):
        for issue in res:
            issue_num = int(issue['url'].split('/')[-1])

            if not issue["labels"]:  # only label unlabelled issues
                label = classify(issue['title'], issue['body'])
                add_label(issue_num, label)
                print(f"Classified issue {issue_num} as {label}.")


if __name__ == "__main__":
    main()
