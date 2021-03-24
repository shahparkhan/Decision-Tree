# Decision-Tree
Python3 code to make decision trees in form of python dictionary


## Pre-requisites:
encodings.txt: contains integer mapping for labels.
dataset.txt: contains labels for string attribute values. 

## How to run the code
python3 makeDecisionTree.py encodings.txt dataset.txt

## Functions

`def initiator(encodings = None, dataset = None)`
Parses .txt files and passes to decisionTreeLearning.

`def decisionTreeLearning(examples = None, attr = None, parent_examples = None)`
Recurrsively makes decision tree dictionaries and append them together.

`def importance(attr, examples)`
Calculates information gain.

`def entropy(count1, count2)`
Calculates entropy.
