from typing import Dict, List
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time

def get_raw_training_data(filename : str) -> List[Dict[str, str]]: 
    """
    Reads csv data dialogue from characters into a list of input dictionaries
    each of the format {person: [person], sentence: [sentence]}
    """
    training_data = []
    with open('./dialogue_data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            character = row[0].lower()
            dialogue_line = row[1].lower()
            training_data.append({"person" : character, "sentence" : dialogue_line})
    return training_data

def preprocess_words(words : List[str], stemmer) -> List[str]:
    """Stems given list of words and returns list of stemmed words with duplicates removed."""
    stems = set()
    for w in words:
        stems.add(stemmer.stem(w))
    return list(stems)

def organize_raw_training_data(raw_training_data : List[Dict], stemmer) -> (List, List, Dict):
    """
    Returns several organized versions of words from character dialogue.
    params: 
        raw_training_data -- format {person: [person], sentence: [sentence]}
    returns: 
        words -- list of all unique words
        classes -- list of all unique characters with lines 
        document -- list of (character, [all unique spoken words])
    """
    words = set() # ["bla", "sldj", "aklj", "woeiu", "lkajdslfsj", "lsdkjfsl", "alskd"]
    document_dict = {} # {char1 : [{"bla", "sldj", "aklj"}, {"asd", "afs"}]}
    documents = [] # [(char1, ["bla", "sldj", "aklj"]), (char1, ["aklj"]), (char2, ["wiu", "lk"])]
    classes = [] # [char1, char2]

    # Process words
    for line in raw_training_data:
        character = line["person"]
        line_tokens = nltk.word_tokenize(line["sentence"]) 
        line_words = preprocess_words(line_tokens, stemmer)
        if character not in classes:
            classes.append(character)
            document_dict.update({character : [set(line_words)]})
        else: # character has already appeared
            word_set = document_dict[character]
            word_set.append(set(line_words))

    # Create output document list
    for character, sentence_words in document_dict.items():
        for w in sentence_words:
            documents.append((list(w), character))
            words.update(w)
    return list(words), classes, documents

def create_training_data(words : List, classes : List, documents: List, stemmer) -> (List, List):
    """
    Converts data from words, classes and documents into binary list representations. 
    params:
        words -- list of all unique words
        classes -- list of all unique characters with lines 
        document -- list of ([unique words in sentence], character)
    returns: 
        training_data -- represents each sentence as its coverage of entire word list,
        where '1' is a word in the sentence and '0' is a word not in the sentence. 
        output -- represents a sentence as what class it belongs to. A sentence that belongs 
        to the zeroth class would be [1, 0, 0]
    """
    training_data = []
    output = []

    for doc in documents:
        sentence_words = doc[0]
        char = doc[1]
        binary_words = binary_list(words, sentence_words)
        binary_classes = binary_list(classes, char)
        training_data.append(binary_words)
        output.append(binary_classes)
    return training_data, output

def binary_list(superset : List, subset : List) -> List:
    """
    Returns a binarized representation of the superset. Elements in the result are 0
    if not included in the subset and 1 if in the subset.
    """
    result = [0] * len(superset)
    subset_elements = set(subset)
    for i in range(0, len(superset)):
        if superset[i] in subset_elements:
            result[i] = 1
    return result


def main():
    stemmer = LancasterStemmer()
    print("getting raw data...")
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    print("organizing data...")
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    # print(f"words={words}")
    # print(f"classes={classes}")
    # print(f"documents={documents}")
    training_data, output = create_training_data(words, classes, documents, stemmer)
    # print(f"training data={training_data}")
    # print(f"output={output}")
    
if __name__ == "__main__":
    main()