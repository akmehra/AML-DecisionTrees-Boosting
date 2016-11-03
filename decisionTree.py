# Author: Mohit Galvankar
# Builds decision tree and classifies input data based on the tree generated
# Run from shell: python DecisionTree.py <train dataset .arff> <test dataset .arff> <depth>
# Takes arff file as input
# b



from __future__ import division
import math
import operator
# import arff
import sys
import DataProcessing


# from tabulate import tabulate


class DNode:
    def __init__(self, feature=None, value=None, result=None, left=None, right=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.left = left
        self.right = right


# Main fucntion
# Computes best information gain by iterating through all the features. Split data on the best info gain and creates tree nodes.
def calculate(data, userDepth, depth):
    # If the data is null return new noxde
    if len(data) == 0:
        return DNode()

    if len(set([x[0] for x in data])) <= 1:
        return DNode(result=results(data))

    if depth < userDepth:
        bestGain, infoGain, bestFeature, bestValue = 0.0, 0.0, 0, 0

        entropyData = entropy_main(data)

        for feature in range(1, len(data[0])):
            uniqueDataPoints = list(set([x[feature] for x in data]))
            for uniqueDataPoint in uniqueDataPoints:
                infoGain, dLeft, dRight = splitdata(entropyData, data, feature, uniqueDataPoint)
                if infoGain > bestGain:
                    bestFeature, bestValue, bestGain, bestDLeftNode, bestDRightNode \
                        = feature, uniqueDataPoint, infoGain, dLeft, dRight

        if bestGain > 0 and len(bestDLeftNode) > 0 and len(bestDRightNode) > 0:
            dLeftNode = calculate(bestDLeftNode, userDepth, depth + 1)
            dRightNode = calculate(bestDRightNode, userDepth, depth + 1)

            return DNode(feature=bestFeature, value=bestValue, left=dLeftNode, right=dRightNode)

    return DNode(result=results(data))


def results(data):
    dict = {}
    for row in data:
        dict.setdefault(row[0], 0)
        dict[row[0]] += 1
    return max(dict.iteritems(), key=operator.itemgetter(1))[0]


# calculate the entropy
def entropy_main(p):
    pos = sum(1 for x in p if x[0] == 1)
    neg = len(p) - pos

    prob_pos = float(pos / (pos + neg))
    prob_neg = 1.0 - prob_pos

    if prob_pos == 1:
        return 1
    if prob_neg == 1:
        return 1

    return -(prob_neg * math.log(prob_neg, 2)) - (prob_pos * math.log(prob_pos, 2))


def cal_infogain(entparentdata, eright, eleft, lendleft, lendright):
    infogain = entparentdata - (lendleft / (lendleft + lendright)) * eleft - (lendright / (
    lendleft + lendright)) * eright
    return infogain


def splitdata(entropy, data, feature, uniqueDataPoint):
    dleft, dright = [], []

    [dleft.append(i) if i[feature] == uniqueDataPoint else dright.append(i) for i in data]

    if len(dright) > 0:
        entright = entropy_main(dright)
    else:
        entright = 0

    if len(dleft) > 0:
        entleft = entropy_main(dleft)
    else:
        entleft = 0

    infogain = cal_infogain(entropy, entright, entleft, len(dleft), len(dright))

    return infogain, dleft, dright


def printtree(tree, indent=''):
    if tree.result != None:
        print ("Result", str(tree.result))
    else:
        print ("If Feature ", str(tree.feature) + ' and Value ' + str(tree.value) + " :")
        print(indent + 'Tree left->')
        printtree(tree.left, indent + '  ')
        print(indent + 'Tree right->')
        printtree(tree.right, indent + '  ')


def classify(tree, datapoint):
    if (tree.result != None):
        return tree.result

    feature = tree.feature
    value = tree.value
    if value == datapoint[feature]:
        label = classify(tree.left, datapoint)
    else:
        label = classify(tree.right, datapoint)

    return label


def classify_accu(tree, tdata):
    count = 0
    for i in tdata:
        predicted = classify(tree, i)
        # print "predicted for",i,"is",predicted
        solution = i[0]
        if predicted == solution:
            count = count + 1
    accuracy = count / len(tdata)
    return accuracy


def compute_confmatrix(tree, tdata):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    n = len(tdata)
    for i in tdata:
        predicted = classify(tree, i)
        # print "predicted for",i,"is",predicted
        solution = i[0]
        if predicted == 1 and solution == 1:
            TP = TP + 1
        elif predicted == 0 and solution == 0:
            TN = TN + 1
        elif predicted == 0 and solution == 1:
            FN = FN + 1
        elif predicted == 1 and solution == 0:
            FP = FP + 1
    confusion_matrix = [[TN, FN], [FP, TP]]
    # print confusion_matrix
    error = (FN + FP) / (n)
    print("Error ", error)
    print("Confusion Matrix :")
    for i in confusion_matrix:
        print (i)
        # print tabulate([['Actual : No', TN, FP], ['Actual : Yes', FN,TP]], headers=[' N : %s' %(n),'Predicted : No', 'Predicted : Yes'],tablefmt='orgtbl')

if __name__ == "__main__":
    solution = []
    f = open(sys.argv[1], 'r')
    data = DataProcessing.preprocess(f)

    t = open(sys.argv[2], 'r')
    tdata = DataProcessing.preprocess(t)

    user_depth = int(sys.argv[3])

    tree = calculate(data, user_depth, 0)

    confusion_matrix = []

    compute_confmatrix(tree, tdata)

    accuracy = classify_accu(tree, tdata)

    print ("Accuracy : ", accuracy)
