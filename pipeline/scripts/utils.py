#!/usr/bin/env python.

import re
import numpy as np


def has_log(text):
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:|\
        [Tt]raceback.{0,10}:|[Bb]acktrace.{0,10}:|[Ll]ogs?:|\d{2}:\d{2}:\d{2}|\
        (INFO|[Ii]nfo|FAIL|[Ff]ail|WARN(ING)?|[Ww]arn(ing)?|FATAL|[Ff]atal|DEBUG|[Dd]ebug|SYSTEM|[Ss]ystem|ERROR|[Ee]rror)\s*:'
    results = re.findall(pattern, text)
    return len(results) != 0


def remove_log(text):
    # If logs appear in a code block, remove the whole code block
    CODE_REGEX = r'```.+?```'
    for match in re.findall(CODE_REGEX, text, flags=re.S):
        if has_log(str(match)):
            text = text.replace(str(match), '')

    # If logs appear outside a code block, remove corresponding lines
    LOG_REGEX = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:|\
        [Tt]raceback.{0,10}:|[Bb]acktrace.{0,10}:|[Ll]ogs?:|\d{2}:\d{2}:\d{2}|\
        (INFO|[Ii]nfo|FAIL|[Ff]ail|WARN(ING)?|[Ww]arn(ing)?|FATAL|[Ff]atal|DEBUG|[Dd]ebug|SYSTEM|[Ss]ystem|ERROR|[Ee]rror)\s*:).+?\n'
    for match in re.findall(LOG_REGEX, text, flags=re.S):
        text = text.replace(str(match), '')

    return text


def has_code_block(text):
    CODE_REGEX = r'```.+?```'
    for match in re.findall(CODE_REGEX, text, flags=re.S):
        if not has_log(str(match)):
            return True

    return False


def remove_code_block(text):
    CODE_REGEX = r'```.+?```'
    for match in re.findall(CODE_REGEX, text, flags=re.S):
        if not has_log(str(match)):
            text = text.replace(str(match), '')
    return text


def has_url(text):
    pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    results = re.findall(pattern, text)
    return len(results) != 0


def remove_url(text):
    # remove link in markdown
    markdown_pattern = r'(?:__|[*#])|\[(.*?)\]\(.*?\)'
    replacement_alt_text = r'\1'
    result_1 = re.sub(markdown_pattern, replacement_alt_text, text)

    # remove link in plain text
    plain_pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    replacement_plain = ''
    result_2 = re.sub(plain_pattern, replacement_plain, result_1)
    return result_2


def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text


def average_results(results):
    """ Takes in list of dictionary result and returns the average dictionary results. """
    # combine results
    avg_results = {}
    for k in results[0].keys():
        from functools import reduce
        avg_results[k] = reduce(lambda a, b: a + b[k], results, 0) / len(results)

    return avg_results


def accuracy_labelled(pred, y_test):
    assert len(pred) == len(y_test), "Wrong dimensions!"
    total_labelled_instances = 0
    total_correct_instances = 0
    num_doc = 0
    for i in range(len(pred)):
        if pred[i] != -1:
            total_labelled_instances += 1
            if pred[i] == y_test[i]:
                total_correct_instances += 1
            if pred[i] == 2:
                num_doc += 1

    # The following stats are just FYI and for checking purposes
    # print("total is ", total_labelled_instances)
    # print("total correct is ", total_correct_instances)
    # print("number of doc issues: ", num_doc)

    return total_correct_instances / total_labelled_instances


def test_average_results():
    results = [
        {'eval_loss': 0.37910524010658264, 'eval_accuracy': 0.8940754039497307, 'eval_precision': 0.8913983988380442,
         'eval_recall': 0.8940754039497307, 'eval_fscore': 0.8921360591455132, 'eval_cm': np.array([[1260, 208, 8],
                                                                                                    [140, 2760, 14],
                                                                                                    [13, 30, 66]]),
         'eval_runtime': 37.5221, 'eval_samples_per_second': 103.912, 'eval_mem_cpu_alloc_delta': 129298,
         'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 339541, 'eval_mem_gpu_peaked_delta': 289573376,
         'dataset size': 3899},
        {'eval_loss': 0.37910524010658264, 'eval_accuracy': 0.8940754039497307, 'eval_precision': 0.8913983988380442,
         'eval_recall': 0.8940754039497307, 'eval_fscore': 0.8921360591455132, 'eval_cm': np.array([[660, 208, 8],
                                                                                                 [140, 2760, 14],
                                                                                                 [13, 30, 66]]),
         'eval_runtime': 37.5221, 'eval_samples_per_second': 103.912, 'eval_mem_cpu_alloc_delta': 129298,
         'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 339541, 'eval_mem_gpu_peaked_delta': 289573376,
         'dataset size': 3899}]
    print(average_results(results))


def tests():
    test_average_results()


if __name__ == "__main__":
    tests()
