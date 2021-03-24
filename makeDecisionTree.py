import pandas as pd
import numpy as np 
import sys
from os import path
from pprint import pprint
from random import choice

encodings_dict = {}

def entropy(count1, count2):
    if count1 == count2:
        return 1.0
    if count1 == 0 or count2 == 0:
        return 0.0
    prob1 = float(count1)/float(count1+count2)
    prob2 = float(count2)/float(count1+count2)
    return (-(prob1*np.log2(prob1)) - (prob2*np.log2(prob2)))

def importance(attr, examples):
    entropy_before = entropy(examples["play"].tolist().count(0), examples["play"].tolist().count(1))

    entropy_after = {}

    for val in attr:
        attr_vals = list(np.unique(examples[val].tolist()))
        total_count = len(examples["play"].tolist())
        entropy_list = []
        for attr_val in attr_vals:
            exs = examples[examples[val] == attr_val]
            ent = (len(exs["play"].tolist())/total_count)*entropy(exs["play"].tolist().count(0), exs["play"].tolist().count(1))
            entropy_list.append(ent)
        entropy_after[val] = sum(entropy_list)
    
    information_gain = {key: entropy_before - entropy_after[key] for key in entropy_after}
    for key, value in information_gain.items():
        if np.isnan(value):
            information_gain[key] = 0.0
    return information_gain

def decisionTreeLearning(examples = None, attr = None, parent_examples = None):
    
    if len(np.unique(examples["play"].tolist())) == 1:
        if examples["play"].iloc[0] == 0:
            return "No"
        else:
            return "Yes"

    else:
        importances = importance(attr, examples)
        max_attr = max(importances, key = importances.get)
        tree = {}

        for val in list(set(examples[max_attr].tolist())):
            ex = examples[examples[max_attr] == val]
            temp_attr = []
            for attribute in attr:
                if attribute != max_attr:
                    temp_attr.append(attribute)
            subtree = decisionTreeLearning(ex, temp_attr, examples)
            tree[encodings_dict[max_attr][val]] = subtree
    
    return {max_attr: tree}

def initiator(encodings = None, dataset = None):
    lines = []
    with open(f'./{encodings}', 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip().split(','), lines))
        for i in range(len(lines[0])):
            encodings_dict[lines[0][i]] = lines[i+1]
    examples = pd.read_csv(f'./{dataset}', delimiter = ",", header = None, names = list(encodings_dict.keys()), dtype = np.int64)

    pprint(decisionTreeLearning(examples = examples, attr = list(encodings_dict.keys())[:-1], parent_examples = examples))
   
if __name__ == "__main__":
    try:
        if (len(sys.argv) != 3):
            print(f'\nIncorrect number of inputs\ntry this: makeDecisionTree.py encodings.txt dataset.txt\n\nShutting Down...')
            sys.exit(0)

        elif not path.exists(sys.argv[1]):
            print(f'\n{sys.argv[1]} does not exist in your directory\nShutting Down...')
            sys.exit(0)

        elif not path.exists(sys.argv[2]):
            print(f'\n{sys.argv[2]} does not exist in your directory\nShutting Down...')
            sys.exit(0)
    
        else:
            initiator(sys.argv[1], sys.argv[2])
    
    except KeyboardInterrupt:
        print(f'\nUnexpected Error: {KeyboardInterrupt}.\nShutting Down...')
        
        sys.exit(0)