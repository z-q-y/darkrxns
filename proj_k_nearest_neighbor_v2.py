import os
os.chdir(r'D:\Haverford\2017-2018\Chem 362')

import pickle
from Python_Exercise.linear_algebra import distance
from collections import Counter

with open('cleandata.p', 'rb') as g:
    rawdata, reality, Zscore_exempted_cols, coldict, rowdict = pickle.load(g)

domain = set(range(1,2000))
trainingrows = set(range(1, 1500))
validationrows = set()
testingrows = domain.difference(trainingrows.union(validationrows))

train_rawdata = [rawdata[index - 1] for index in trainingrows]
test_rawdata = [rawdata[index - 1] for index in testingrows]
train_reality = [reality[index - 1] for index in trainingrows]
test_reality = [reality[index - 1] for index in testingrows]

traindata = list(zip(train_rawdata, train_reality))
testdata = list(zip(test_rawdata, test_reality))

""" When validation becomes necessary
with open('validationset.p', 'rb') as g3:
    validationdata = pickle.load(g3)
"""

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count 
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest
    
def knn_classify(k, labeled_points, new_point):
    # each labeled point should be a pair (point, label)
    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda args: distance(args[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)

comparison = []

if __name__ == "__main__":

    # try several different values for k
    for k in [1, 3, 5, 7]:
        num_correct = 0

        for dataset, result in testdata:
            predicted_result = knn_classify(k, traindata, dataset)            
            comparison.append([predicted_result, result]) # Make a list of all the outcomes
            
            if predicted_result == result:
                num_correct += 1

        print(k, "neighbor[s]:", num_correct, "correct out of", len(testdata))
        print(comparison)
            
# Change size of training and test set and make learning curve out of it