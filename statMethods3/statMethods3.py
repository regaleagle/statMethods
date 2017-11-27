from __future__ import division
from codecs import open
import pandas as pd
import numpy as np
from collections import Counter
from operator import itemgetter
import scipy.stats as stats

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
    return max(prob_list, key=itemgetter(1))

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

def incorrectly_labeled_documents(true_labels, guessed_labels, documents):
    incorrect_docs = []

    for i in range(len(true_labels)):
        if true_labels[i] != guessed_labels[i][0]:
            incorrect_docs.append((documents[i], guessed_labels[i][1]))
    
    return incorrect_docs

def cross_validate(N, docs, labels):
    lower_bound = float(0)
    upper_boud = float(0)

    total_correct = 0
    total_out_of = 0

    for fold_nbr in range(N):
        split_point_1 = int(float(fold_nbr)/N*len(docs))
        split_point_2 = int(float(fold_nbr+1)/N*len(docs))

        train_docs_fold = docs[:split_point_1] + docs[split_point_2:]
        train_labels_fold = labels[:split_point_1] + labels[split_point_2:]

        eval_docs_fold = docs[split_point_1:split_point_2]
        eval_labels_fold = labels[split_point_1:split_point_2]

        fold_model = train_nb(train_docs_fold, train_labels_fold)
        fold_guesses = classify_documents(eval_docs_fold, model)

        correct, out_of = accuracy(eval_labels_fold, [guess[0] for guess in fold_guesses])
        total_correct += correct
        total_out_of += out_of

        incorrect = out_of - correct
        posterior_distr = stats.beta(correct + 1, incorrect + 1)
        low, high = posterior_distr.interval(0.95)

        lower_bound += low
        upper_boud += high
        
    return float(lower_bound) / float(N), float(upper_boud) / float(N), total_correct, total_out_of

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

model = train_nb(train_docs, train_labels)

#test small cases
print("dummy case 1 'great','best' ", classify_nb(['great','best'], model)[0])
print("dummy case 2 'bad','worst' ", classify_nb(['bad','worst'], model)[0])

#test against eval

guesses = classify_documents(eval_docs, model)

correct, out_of = accuracy(eval_labels, [x[0] for x in guesses])

print("Test docs held for evaluation:")
print("guessed: ", correct, "out of: ", out_of, " Percent: ", (correct / out_of) * 100)

# Error Analysis - high probability incorrect documents

incorrectly_labelled_log_prob = incorrectly_labeled_documents(eval_labels, guesses, eval_docs)
incorrectly_labelled_prob = sorted([(x[0], np.exp(x[1])) for x in incorrectly_labelled_log_prob], key= lambda x: x[1], reverse=True)

with open('incorrect_docs_with_high_probabilities.txt', 'w+') as f:
    for i in range(len(incorrectly_labelled_prob)):

        il = incorrectly_labelled_prob[i]
        classification = guesses[i]

        f.write("Document:\n{}\nIncorrectly labelled {} with prob: {}\n\n".format(" ".join(il[0]), classification[0], il[1]))

# Interval estimate

incorrect = out_of - correct
posterior_distr = stats.beta(correct + 1, incorrect + 1)
print('95% credibility interval: {}'.format(posterior_distr.interval(0.95)))

# Cross validation

lower, upper, cross_validation_correct, cross_validation_out_of = cross_validate(10, all_docs, all_labels)
print('Cross validation 95% credibility interval: {}, {}'.format(lower, upper))

# Comparing accuracy to a target value

p_val = stats.binom_test(cross_validation_correct, n=cross_validation_out_of, p=0.80)
print('The p-value of the binomial hypothesis test: {}'.format(p_val))
# Given that the p-value is much smaller than 0.05, we can assume that the null hypothesis (the accuracy of the model > 0.8) holds true.


