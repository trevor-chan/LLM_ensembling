import random
import time
from multiprocessing import Pool
import numpy as np
import json
import os

def calc_mode(responses):
    mode = []
    max_count = responses.count(responses[0])
    for response in set(responses):
        if responses.count(response) > max_count:
            mode = [response]
            max_count = responses.count(response)
        elif responses.count(response) == max_count:
            mode.append(response)
    return mode, max_count

def vote(responses, threshold, correct):
    mode, count = calc_mode(responses)
    if len(mode) > 1:
        return 0
    elif count/len(responses) < threshold:
        return 0
    elif mode[0] == correct:
        return 1
    else:
        return -1

def vote_count(responses):
    return [int(np.sum(np.array(responses) == i)) for i in [-1, 0, 1]]

def calc_entropy(responses):
    entropy = []
    for response in responses:
        counts = np.array([response.count(i) for i in set(response)])
        probs = counts/np.sum(counts)
        entropy.append(-np.sum(probs*np.log(probs)))
    return entropy

def normalized_entropy(responses):
    entropy = []
    for response in responses:
        counts = np.array([response.count(i) for i in set(response)])
        probs = counts/np.sum(counts)
        h = -np.sum(probs*np.log(probs))
        entropy.append(h / np.log(len(response)))
    return entropy

def simple_entropy(responses):
    responses = np.array(responses).T
    return [(len(np.unique(responses[:,i])) - 1) / len(responses) for i in range(responses.shape[1])]

def fmax(responses, difficulty):
    uncertainty = np.array(simple_entropy(responses))
    confusion = (np.array(difficulty) - np.array(uncertainty))/(1 - np.array(uncertainty))
    fmax = (1-uncertainty)*(np.array([max(c, 1-c) for c in confusion]))
    return fmax