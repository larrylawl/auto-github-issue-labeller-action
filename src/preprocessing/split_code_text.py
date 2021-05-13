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

    data = json.load(open(file_path, 'r'))
    
    arr = []

    for post in data:
        dic = {'codeblocks': [], 'text': ""}
        
        body = post['body']
        CODE_REGEX = r'```.+?```'
        for match in re.findall(CODE_REGEX, body, flags=re.S):
            dic['codeblocks'].append(str(match))

        dic['text'] = re.sub(CODE_REGEX, '', body, flags=re.S)
        dic['title'] = post['title']
        dic['labels'] = post['labels']
        
        arr.append(dic)

    with open(f'{file_name}_text_code_split.json', 'w+') as f:
        f.write(json.dumps(arr, indent=4))

