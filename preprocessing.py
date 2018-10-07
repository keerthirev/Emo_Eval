import spacy
import json
from pprint import pprint
import re

##
# Takes in a path to training data, and returns two
# arrays: for every sample in x_train, the corresponding
# index in y_train contains the correct label
##
def readData(path):
    # Read in the training data
    with open(path) as fp:
        data = [x.split('\t') for x in fp.readlines()[1:]]

    # Populate x_train with the conversation
    # snippets and y_train with the label
    x_train = []
    y_train = []
    for sample in data:
        x_train.append(sample[1:4])
        y_train.append(sample[4])

    return x_train, y_train

##
# 
##
def preprocessing(data):
    return data


def main():
    nlp = spacy.load('en')
    nlp.max_length

    x_train, y_train = readData('train.txt')
    # data = preprocessing(x_train)

    doc = nlp('\n'.join(['.'.join(x) for x in x_train][:5000]))

    # for token in doc:
    #     print(token)
    
    spacy.displacy.serve(doc, style='dep')


if __name__ == '__main__':
    main()
