import sys
import json
import re
import os

import codecs

files = list(sys.argv[1:])

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

for file_path in files:
    base = os.path.basename(file_path)
    file_name = os.path.splitext(base)[0]

    data = json.load(open(file_path, 'r', encoding='utf-8'))
    
    bad_ids = set()

    with open(f'bad_ids/{file_name}_bad_ids.txt', 'r') as f:
        for line in f:
            bad_ids.add(int(line.strip()))

    arr = []

    for post in data:
        dic = {}
        dic['body'] = post['body']
        dic['title'] = post['title']
        dic['labels'] = post['labels']
        dic['id'] = post['id']

        if dic['id'] not in bad_ids:
            arr.append(dic)

    with open(f'{file_name}_removed_dups.json', 'w+') as f:
        f.write(json.dumps(arr, indent=4))

