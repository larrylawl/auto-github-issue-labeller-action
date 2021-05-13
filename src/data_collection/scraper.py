import json
import requests
import sys
import time
from dotenv import load_dotenv
import os

arr = []

for arg in sys.argv[1:]:
    with open(arg, 'r') as f:
        j = json.load(f)
        arr.append(j)

load_dotenv()
SECRET_TOKEN = os.environ.get("GITHUB_TOKEN")

headers = {
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    "Authorization": "token " + SECRET_TOKEN,
}

payload = {
    "state": "all",
    "per_page": 100,
}

s = requests.Session()

desired_inputs = ['title', 'body']

rate_left = int(s.get('https://api.github.com/rate_limit').json()['rate']['remaining'])

for dic in arr:
    repo = dic['repo']
    print(f"Scraping {repo}...")

    target_labels = dic['desired_labels']

    if len(target_labels) == 0:
        target_labels = [None]

    user, repo = repo.split('/')
    arr = []
    for label in target_labels:
        page = 0
        if label:
            print(label)
            payload['labels'] = label

        while True:
            if rate_left == 0:
                print("Used up limit. Sleeping...")
                time.sleep(300)
                rate_left = int(s.get('https://api.github.com/rate_limit').json()['rate']['remaining'])
                continue
            payload['page'] = page
            res = s.get(f'https://api.github.com/repos/{user}/{repo}/issues', headers=headers, params=payload)
            print("requesting page", page)
            if len(res.json()) == 0:
                if page == 0:
                    print(f"Warning: {label} returned no pages")
                break
            for dic in res.json():
                if 'pull_request' not in dic:
                    data = {field: dic[field] for field in desired_inputs}
                    data['labels'] = label 
                    arr.append(data)
            page += 1
            rate_left -= 1

    with open(f'{repo}.json', 'w+') as f:
        f.write(json.dumps(arr, indent=4))
