#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 03:10:57 2021

@author: Hotness
"""

import math
import sys
from collections import Counter

#admin stuff
training_data = [line.rstrip().split() for line in open(sys.argv[1]).readlines()]
test_data = [line.rstrip().split() for line in open(sys.argv[2]).readlines()]
class_prior_delta = float(sys.argv[3])
cond_prob_delta = float(sys.argv[4])
model_file = sys.argv[5]
sys_output = sys.argv[6]

#Creating training and test sets
training_labels = []
training_vectors = []
test_labels = []
test_vectors = []

word_counts = Counter() #This will help us calculate the probability of each feature in a minute
label_counts = Counter() #This will store our label counts
word_counts_per_label = {} # This will store our feature counts per label

#Vector structure: (label, set(features))
for vector in training_data:
    training_vectors.append((vector[0],set(word.split(':')[0] for word in vector[1:])))
    label_counts.update([vector[0]])
    training_labels.append(vector[0])

for vector in test_data:
    test_vectors.append((vector[0],set(word.split(':')[0] for word in vector[1:])))
    test_labels.append(vector[0])
    
vocab = set()
#test_features = set()
for vector in training_vectors:
    vocab.update(vector[1])
    word_counts.update(vector[1])
    #The next three lines split the feature counts by label
    if vector[0] not in word_counts_per_label:
        word_counts_per_label[vector[0]] = Counter() #Should be a set?  See green text below.
    word_counts_per_label[vector[0]].update(vector[1])
    
vocab_length = len(vocab)

#Training phase
label_sum = sum(label_counts.values())
sum_per_label = {label:sum(word_counts_per_label[label].values()) for label in word_counts_per_label}
label_probs = {label:(label_counts[label]/label_sum) for label in label_counts}
#This will store our feature probs per label
word_probs_per_label = {}
for word in vocab:
    word_probs_per_label[word] = {}
    for label in word_counts_per_label:
        word_probs_per_label[word][label] = math.log10((word_counts_per_label[label][word] + cond_prob_delta) /
                                                          (label_counts[label] + 2 * cond_prob_delta))
                                                                      

def classify_NB(document, vocab, model, confusion_matrix):
    log_classifications = {label:0 for label in word_counts_per_label}
    classifications = {}
    for word in vocab:
        if word not in document[1]:
            for label in word_probs_per_label[word]:
                log_classifications[label] += math.log10(1 - 10 ** word_probs_per_label[word][label])
        else:
            for label in word_probs_per_label[word]:
                log_classifications[label] += word_probs_per_label[word][label]

    largest_log = max(log_classifications.items(), key=lambda x: x[1])[1]
    for label in log_classifications:
        classifications[label] = 10 ** (log_classifications[label] - largest_log)
    prediction = max(classifications.items(), key=lambda x: x[1])[0]
    if document[0] in confusion_matrix:
        confusion_matrix[document[0]][prediction] += 1
    return (classifications, log_classifications, prediction)

#Classify training data and build confusion matrix
training_confusion_matrix = {'talk.politics.guns': {'talk.politics.guns':0, 'talk.politics.mideast':0, 'talk.politics.misc':0},
                    'talk.politics.mideast': {'talk.politics.guns':0, 'talk.politics.mideast':0, 'talk.politics.misc':0},
                    'talk.politics.misc': {'talk.politics.guns':0, 'talk.politics.mideast':0, 'talk.politics.misc':0}}
train_results = [classify_NB(document, vocab, word_probs_per_label, training_confusion_matrix) for document in training_vectors]

training_accuracy = 0
for key in training_confusion_matrix:
    training_accuracy += training_confusion_matrix[key][key]
training_accuracy /= len(training_vectors)

#Classify test data and build confusion matrix
test_confusion_matrix = {'talk.politics.guns': {'talk.politics.guns':0, 'talk.politics.mideast':0, 'talk.politics.misc':0},
                    'talk.politics.mideast': {'talk.politics.guns':0, 'talk.politics.mideast':0, 'talk.politics.misc':0},
                    'talk.politics.misc': {'talk.politics.guns':0, 'talk.politics.mideast':0, 'talk.politics.misc':0}}
test_results = [classify_NB(document, vocab, word_probs_per_label, test_confusion_matrix) for document in test_vectors]

test_accuracy = 0
for key in test_confusion_matrix:
    test_accuracy += test_confusion_matrix[key][key]
test_accuracy /= len(test_vectors)

#Build model file
label_probs_print = '\n'.join(['{} {} {}'.format(label,
                                                 label_probs[label],
                                                 math.log10(label_probs[label])) for label in sorted(label_probs.keys())])   

with open(model_file,'w') as f:
    f.write('%%%%% prior prob P(c) %%%%%\n')
    f.write(label_probs_print + '\n')
    f.write('%%%%% conditional prob P(f|c) %%%%%\n')
    for label in sorted(label_probs.keys()):
        f.write('%%%%% conditional prob P(f|c) c={} %%%%%\n'.format(label))
        for word_label_prob in sorted(word_probs_per_label.items(),key=lambda x:x[0]):
            f.write(('{}\t{}\t{}\t{}\n').format(word_label_prob[0],
                                                label,
                                                10 ** word_label_prob[1][label],
                                                word_label_prob[1][label]))
        f.write('\n')

#Build sys output file
train_sys_print = []
for i,result in enumerate(train_results):
    probs = sorted(result[0].items(),key=lambda x:x[1],reverse=True)
    result = ' '.join(['{} {}'.format(prob[0], prob[1]) for prob in probs])
    train_sys_print.append('array:{} {}'.format(i,result))

test_sys_print = []
for i,result in enumerate(test_results):
    probs = sorted(result[0].items(),key=lambda x:x[1],reverse=True)
    result = ' '.join(['{} {}'.format(prob[0], prob[1]) for prob in probs])
    test_sys_print.append('array:{} {}'.format(i,result))

with open(sys_output,'w') as f:
    f.write('%%%%% training data:\n')
    f.write('\n'.join(train_sys_print))
    f.write('\n')
    f.write('\n')
    f.write('%%%%% test data:\n')
    f.write('\n'.join(test_sys_print))

#Print training confusion matrix and accuracy
print('Confusion matrix for the training data:')
print('row is the truth, column is the system output')
print()
train_keys = training_confusion_matrix.keys()
print('\t' + ' '.join(train_keys))
for outer_key in training_confusion_matrix:
    train_counts = []
    for inner_key in train_keys:
        train_counts.append(str(training_confusion_matrix[outer_key][inner_key]))
    print (outer_key + ' ' + ' '.join(train_counts))
print()
print('  Training accuracy={}'.format(training_accuracy))
print()
    
#Print training confusion matrix and accuracy
print('Confusion matrix for the test data:')
print('row is the truth, column is the system output')
print()
test_keys = test_confusion_matrix.keys()
print('\t' + ' '.join(test_keys))
for outer_key in test_confusion_matrix:
    test_counts = []
    for inner_key in test_keys:
        test_counts.append(str(test_confusion_matrix[outer_key][inner_key]))
    print (outer_key + ' ' + ' '.join(test_counts))
print()
print('  Test accuracy={}'.format(test_accuracy))