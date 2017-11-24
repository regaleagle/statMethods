from __future__ import division
from codecs import open
import pandas as pd
import numpy as np
from collections import Counter
from operator import itemgetter

UNKNOWN = "unknown_keyword"

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.replace(',', '').replace('.', '').strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


def train_nb(documents, labels):
    #bag the words
    label_bag_words = {}
    for doc, label in zip(documents, labels):
        if label in label_bag_words:
            label_bag_words[label].extend(doc)
        else:
            label_bag_words[label] = doc

    #use frequencys and total word count to get log probability
    freq_dict = {}
    for uniq_label in set(labels):
        total_freq = len(label_bag_words[label])
        counts = Counter(label_bag_words[uniq_label])
        probability_dict = {UNKNOWN:1.0/float(total_freq + len(counts.keys()))}
        for key, value in counts.items():
            probability_dict[key] = float(value + 1) / float(total_freq + len(counts.keys()))
        freq_dict[uniq_label] = probability_dict

    return freq_dict

def compute_logprob(document, label, model):
    probability = 0
    for word in document:
        if word in model[label]:
            probability += np.log(model[label][word])
        else:
            probability += np.log(model[label][UNKNOWN])
    return probability

def classify_nb(document, model):
    prob_list = []
    for label in model.keys():
        prob_list.append((label, compute_logprob(document, label, model)))
    return max(prob_list, key=itemgetter(1))[0]

def classify_documents(docs, model):
    labels = []
    for doc in docs:
        labels.append(classify_nb(doc, model))
    return labels

def accuracy(true_labels, guessed_labels):
    count_correct = 0
    for true_lab, guessed_lab in zip(true_labels, guessed_labels):
        if true_lab == guessed_lab:
            count_correct += 1
    return count_correct, len(true_labels)

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

model = train_nb(train_docs, train_labels)

#test small cases
print("dummy case 1 'great','best' ", classify_nb(['great','best'], model))
print("dummy case 2 'bad','worst' ", classify_nb(['bad','worst'], model))

#test against eval

guesses = classify_documents(eval_docs, model)

correct, out_of = accuracy(eval_labels, guesses)

print("Test docs held for evaluation:")
print("guessed: ", correct, "out of: ", out_of, " Percent: ", (correct / out_of) * 100)