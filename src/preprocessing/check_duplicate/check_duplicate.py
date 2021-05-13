import sys
import json
import re
import os

import codecs

whitelist = set(["type:feature", "type:bug","type:docs-feature", "type:docs-bug","C-feature-request", "C-feature-accepted", "C-enhancement","C-bug","T-doc","kind/feature", "kind/api-change","kind/bug","kind/documentation",'severe: new feature',"severe: crash", "severe: fatal crash", "severe: rendering","documentation","Feature", "Enhancement","Bug","Type: documentation","enhancement :sparkles:","bug :beetle:", "crash :boom:","documentation :notebook:"])


files = list(sys.argv[1:])

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

for file_path in files:
    base = os.path.basename(file_path)
    file_name = os.path.splitext(base)[0]

    data = json.load(open(file_path, 'r', encoding='utf-8'))
    
    dic = {}
    duplicated_keys = set()
    for post in data:
        key = (post['body'], post['title'])
        post_id = post['id']
        label = post['labels']

        if label not in whitelist:
            continue

        if key not in dic:
            dic[key] = [(post_id, label)]
        else:
            duplicated_keys.add(key)
            dic[key].append((post_id, label))
    
    print(base)
    
    bad_ids = []

    for key in duplicated_keys:
        labels = [label for _, label in dic[key]]
        ids = [i for i, _ in dic[key]]

        if all([l == labels[0] for l in labels]):
            bad_ids.append(ids[0])
        else:
            bad_ids.extend(ids)
    bad_ids = sorted(set(bad_ids))


    with open(f'{file_name}_bad_ids.txt', 'w+') as f:
        for i in bad_ids:
            f.write(f'{i}\n')
