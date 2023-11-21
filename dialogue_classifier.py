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
        document -- list of (character, [all unique spoken words])
    returns: 
        training_data -- represents each sentence as its rcoverage of entire word list,
        where '1' is a word in the sentence and '0' is a word not in the sentence. 
        output -- represents a sentence as what class it belongs to. A sentence that belongs 
        to the zeroth class would be [1, 0, 0]
    """
    training_data = []
    output = []
    # populate training data
    # populate output data
    pass 

def main():
    stemmer = LancasterStemmer()
    print("getting raw data...")
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    print("organizing data...")
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    print(f"words={words}")
    print(f"classes={classes}")
    print(f"documents={documents}")
    # training_data, output = create_training_data(words, classes, documents, stemmer)

if __name__ == "__main__":
    main()