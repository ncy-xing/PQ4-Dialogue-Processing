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

def organize_raw_training_data(raw_training_data : List[Dict], stemmer) -> None:
    words = [] # ["bla", "sldj", "aklj", "woeiu", "lkajdslfsj", "lsdkjfsl", "alskd"]
    document_dict = {} # {char1 : ["bla", "sldj", "aklj"]}
    documents = [] # [(char1, ["bla", "sldj", "aklj"]), (char2, ["woeiu", "lkajdslfsj", "lsdkjfsl", "alskd"])]
    classes = [] # [char1, char2]
    # Retrieve the tokens from the sentence.  You can achieve this with `nltk.word_tokenize()
    for line in raw_training_data:
        character = line["person"]
        line_tokens = nltk.word_tokenize(line["sentence"]) 
        line_words = preprocess_words(line_tokens, stemmer)
        if character not in classes:
            classes.append(character)
            document_dict.update({character : line_words})
        else: # character has already appeared
            document_dict[character] += (line_words)
        
    # print(f"{character} line_tokens: {line_tokens}")
    # print(f"line_words: {line_words}")
    for character, words in document_dict.items():
        print(f"{character}: {words}")

def main():
    stemmer = LancasterStemmer()
    print("getting raw data...")
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    print("organizing data...")
    organize_raw_training_data(raw_training_data, stemmer)


if __name__ == "__main__":
    main()