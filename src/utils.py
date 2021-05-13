import re

import pandas as pd


def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text


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
